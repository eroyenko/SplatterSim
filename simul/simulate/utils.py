from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

import simul.cnv.gene_expression as gex
import simul.base.splatter as splatter

from simul.patients.dataset import Dataset
from simul.base.config import Bridges, SimCellConfig
import simul.base.utils as utils

import os
import pathlib as pl
import json
import anndata as ad

from scipy.interpolate import CubicSpline
from scipy.stats import norm
from simul.base.config import Proportions

from sympy import symbols, Eq, solve
from simul.base.config import CoefficientCache

from scipy.stats import beta

######## SAVING FUNCTIONS ##############


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


def save_dataset(
    adatas: List[ad.AnnData],
    ds_name: str,
    de_group: pd.DataFrame,
    de_batch: pd.DataFrame,
    gain_expr_full: Dict[str, np.ndarray],
    loss_expr_full: Dict[str, np.ndarray],
    savedir: pl.Path,
    config: SimCellConfig,
) -> None:
    os.makedirs(savedir / ds_name, exist_ok=True)

    with open(savedir / ds_name / "config.json", "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, ensure_ascii=False, default=np_encoder)

    for adata in adatas:
        patname = adata.obs.sample_id[0]
        adata.write(savedir / ds_name / f"{patname}.h5ad")

    truefacsdir = savedir / ds_name / "true_facs"
    os.makedirs(truefacsdir, exist_ok=True)
    de_group.to_csv(truefacsdir / "de-groups.csv")
    de_batch.to_csv(truefacsdir / "de-batch.csv")

    cnvdir = savedir / ds_name / "cnv_effects"
    os.makedirs(cnvdir, exist_ok=True)
    for pat in gain_expr_full:
        pd.Series(gain_expr_full[pat]).to_csv(cnvdir / f"{pat}_gain_effect.csv")
    for pat in loss_expr_full:
        pd.Series(loss_expr_full[pat]).to_csv(cnvdir / f"{pat}_loss_effect.csv")


#################### RUNNING SIMULATION #############################


def break_mean_pc(mean_pc: np.ndarray, full_obs: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
    mean_pc_pp = {}
    idx = 0
    for pat in full_obs:
        mean_pc_pp[pat] = mean_pc[idx : idx + full_obs[pat].shape[0]]
        idx += full_obs[pat].shape[0]
    return mean_pc_pp


def get_facs_matrix(de_pp: Dict[str, np.ndarray], full_obs: pd.DataFrame) -> Dict[str, np.ndarray]:

    de_facs = {pat: [] for pat in full_obs}
    for pat in full_obs:
        listprogram = full_obs[pat].program.ravel()
        for pr in listprogram:
            de_facs[pat].append(de_pp[pr])
        de_facs[pat] = np.array(de_facs[pat])
    return de_facs


def get_facs_matrix_batch(de_pp: Dict[str, np.ndarray], full_obs: pd.DataFrame) -> Dict[str, np.ndarray]:
    de_facs = {pat: np.array([de_pp[pat]] * full_obs[pat].shape[0]) for pat in full_obs}
    return de_facs


def get_mask_high(means_pc: np.ndarray, quantile: float = 0.3):
    avg = means_pc.mean(axis=0)
    qt = np.quantile(avg, quantile)
    return avg > qt


def transform_means_by_facs(
    rng: np.random.Generator,
    config: SimCellConfig,
    full_obs: Dict[str, pd.DataFrame],
    mean_pc_pp: Dict[str, pd.DataFrame],
    group_or_batch: str = "group",
) -> Dict[str, pd.DataFrame]:
    if group_or_batch == "group":
        group_names = config.group_names
        p_de_list = config.p_de_list
        p_down_list = config.p_down_list
        de_location_list = config.de_location_list
        de_scale_list = config.de_scale_list
    else:
        group_names = config.batch_names
        p_de_list = config.pb_de_list
        p_down_list = config.pb_down_list
        de_location_list = config.bde_location_list
        de_scale_list = config.bde_scale_list

    groups_de = splatter.get_groups_de(
        rng=rng,
        group_names=group_names,
        n_genes=config.n_genes,
        p_de_list=p_de_list,
        p_down_list=p_down_list,
        de_location_list=de_location_list,
        de_scale_list=de_scale_list,
    )

    de_pp = {group_names[i]: groups_de[i] for i in range(len(group_names))}
    if group_or_batch == "group":
        de_facs = get_facs_matrix(de_pp=de_pp, full_obs=full_obs)
    else:
        de_facs = get_facs_matrix_batch(de_pp=de_pp, full_obs=full_obs)

    transformed_means = splatter.transform_group_means(means_pp=mean_pc_pp, de_facs=de_facs)

    return transformed_means, de_pp


def get_gain_loss_expr(means: np.ndarray, quantile: float = 0.3) -> Tuple[np.ndarray]:
    # we first select which genes belong to the highly/lowly expressed, as the effect of
    # gains/losses on gene expression depends on the original expression of the gene
    mask_high = get_mask_high(means_pc=means, quantile=0.3)
    # simulate the effect of a gain/loss for a specific gene separately for each patient
    gain_expr = gex.sample_gain_vector(mask_high=mask_high)
    loss_expr = gex.sample_loss_vector(mask_high=mask_high)

    return gain_expr, loss_expr


def transform_malignant_means(
    full_obs: Dict[str, pd.DataFrame],
    transformed_means: Dict[str, pd.DataFrame],
    dataset: Dataset,
    shared_cnv: bool = False,
) -> Tuple[Dict[str, np.ndarray]]:

    cnv_transf_means = {pat: [] for pat in full_obs}

    gain_expr_full, loss_expr_full = {}, {}

    if shared_cnv:
        # the CNV effects will be shared across all patients
        full_gex = np.concatenate(list(transformed_means.values()))
        gain_expr, loss_expr = get_gain_loss_expr(means=full_gex, quantile=0.3)
        gain_expr_full["shared"] = gain_expr
        loss_expr_full["shared"] = loss_expr

    for patient in full_obs:
        df_obs = full_obs[patient].copy()
        mask_malignant = (df_obs.malignant_key == "malignant").ravel()

        df_obs = df_obs.loc[mask_malignant]
        new_means = transformed_means[patient].copy()
        mal_means = new_means[mask_malignant]
        cell_subclones = df_obs.subclone.ravel()

        if not (shared_cnv):
            gain_expr, loss_expr = get_gain_loss_expr(means=mal_means, quantile=0.3)
            gain_expr_full[patient] = gain_expr
            loss_expr_full[patient] = loss_expr

        # retrieve the subclone profiles
        mapping_patients = dataset.name_to_patient()
        patient_subclone_profiles = {
            mapping_patients[patient].subclones[i].name: mapping_patients[patient].subclones[i].profile
            for i in range(len(mapping_patients[patient].subclones))
        }

        cnvmeans = []
        for i, sub in enumerate(cell_subclones):
            subclone_profile = patient_subclone_profiles[sub].ravel()

            mean_gex = mal_means[i]

            mean_gex = gex.change_expression(
                mean_gex,
                changes=subclone_profile,
                gain_change=gain_expr,
                loss_change=loss_expr,
            )
            # we clip the values so that 0 entries become 0.0001. This is because we
            # sample from a gamma distribution at the beginning
            # the % of 0 in the data is small enough that the approximation should be ok
            mean_gex = np.clip(mean_gex, a_min=0.0001, a_max=None)

            cnvmeans.append(mean_gex)

        new_means[mask_malignant] = cnvmeans

        cnv_transf_means[patient] = new_means

    return cnv_transf_means, gain_expr_full, loss_expr_full

######### Assignement of cell cycle phase to cells ############
def assign_cycle_phases(
    config: SimCellConfig,
    full_obs: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, np.ndarray]]:
    """Assign a cell cycle phase to each cell in the dataset.
    Cells are filtered by cell type and then assigned a phase based on the proportions of the cell type.

    Args:
    config (SimCellConfig): The configuration of the simulation.
    full_obs (Dict[str, pd.DataFrame]): The DataFrame containing the cells.

    Returns:
    Dict[str, np.ndarray]: The DataFrame containing the cells with the assigned cell cycle phase.
    """
    
    for patient in full_obs:
        df_obs = full_obs[patient].copy()
        original_order = df_obs.index
        celltypes = config.group_names
        masks = [(df_obs.program == celltype).ravel() for celltype in celltypes]

        df_obs_filtered = [df_obs.loc[mask] for mask in masks]
        df_obs_with_phases = map(lambda df, celltype: df.assign(cell_phase=assign_phases(df, celltype, config.cell_cycle_proportions)), 
                                 df_obs_filtered, celltypes)
        
        df_obs_with_cycle_phases = pd.concat(df_obs_with_phases)
        df_obs_with_cycle_phases = df_obs_with_cycle_phases.reindex(original_order)
        full_obs[patient] = df_obs_with_cycle_phases

    return full_obs

def assign_phases(df: pd.DataFrame, program: str, proportions_config: Proportions) -> np.ndarray:
    program_proportions = proportions_config.get_proportion(program)
    phases = list(program_proportions.keys())
    probabilities = list(program_proportions.values())
    cell_phases = np.random.choice(phases, size=len(df), p=probabilities)
    return cell_phases

######### Transforming mean linked to phase ############
def transform_phase_means(
    phase: str,
    full_obs: Dict[str, pd.DataFrame],
    marker_genes: pd.DataFrame,
    transformed_means: Dict[str, pd.DataFrame],
    bridge_params: Bridges,
) -> Dict[str, pd.DataFrame]:

    phase_transf_means = {pat: [] for pat in full_obs}

    num_s_genes = marker_genes['S'].sum()
    num_g2m_genes = marker_genes['G2M'].sum()
    num_identity_genes = len(marker_genes) - (num_s_genes + num_g2m_genes)

    if phase == 'G1':
        bridge_params_dict = {
            'phase' : phase,
            'S_marker_genes': bridge_params.G1_phase_S_genes,
            'G2M_marker_genes': bridge_params.G1_phase_G2M_genes,
            'identity_genes': bridge_params.G1_phase_identity_genes,
            'num_S_genes': num_s_genes,
            'num_G2M_genes': num_g2m_genes,
            'num_identity_genes': num_identity_genes
        }
        
    elif phase == 'S':
        bridge_params_dict = {
            'phase' : phase,
            'S_marker_genes': bridge_params.S_phase_S_genes,
            'G2M_marker_genes': bridge_params.S_phase_G2M_genes,
            'identity_genes': bridge_params.S_phase_identity_genes,
            'num_S_genes': num_s_genes,
            'num_G2M_genes': num_g2m_genes,
            'num_identity_genes': num_identity_genes
        }
    elif phase == 'G2M':
        bridge_params_dict = {
            'phase' : phase,
            'S_marker_genes': bridge_params.G2M_phase_S_genes,
            'G2M_marker_genes': bridge_params.G2M_phase_G2M_genes,
            'identity_genes': bridge_params.G2M_phase_identity_genes,
            'num_S_genes': num_s_genes,
            'num_G2M_genes': num_g2m_genes,
            'num_identity_genes': num_identity_genes
        }

    S_bridges, G2M_bridges, identity_bridges = generate_bridges(bridge_params_dict)
    phase_transf_means = adjust_means_by_phase(
        phase, 
        full_obs, 
        transformed_means, 
        marker_genes, 
        S_bridges,
        G2M_bridges,
        identity_bridges
    )

    return phase_transf_means

def generate_bridges(bridge_params_dict: dict) -> Tuple[np.ndarray]:
    S_bridges = {}
    G2M_bridges = {}
    identity_bridges = {}

    phase = bridge_params_dict['phase']

    if phase == 'G1':
        S_bridges = generate_combined_bridges(bridge_params_dict, 'num_S_genes', 'S_marker_genes', True)
        G2M_bridges = generate_combined_bridges(bridge_params_dict, 'num_G2M_genes', 'G2M_marker_genes', False)
        identity_bridges = generate_simple_bridges(bridge_params_dict, 'num_identity_genes', 'identity_genes')
    
    elif phase == 'S':
        S_bridges = generate_simple_bridges(bridge_params_dict, 'num_S_genes', 'S_marker_genes')
        G2M_bridges = generate_combined_bridges(bridge_params_dict, 'num_G2M_genes', 'G2M_marker_genes', True)
        identity_bridges = generate_simple_bridges(bridge_params_dict, 'num_identity_genes', 'identity_genes')

    elif phase == 'G2M':
        S_bridges = generate_combined_bridges(bridge_params_dict, 'num_S_genes', 'S_marker_genes', False)
        G2M_bridges = generate_simple_bridges(bridge_params_dict, 'num_G2M_genes', 'G2M_marker_genes')
        identity_bridges = generate_simple_bridges(bridge_params_dict, 'num_identity_genes', 'identity_genes')

    return S_bridges, G2M_bridges, identity_bridges

def generate_simple_bridges(bridge_params_dict: dict, num_genes: str, genes: str) -> np.ndarray:
    simple_bridges = utils.generate_valid_bridges(
        bridge_params_dict[num_genes], 
        bridge_params_dict[genes].start_range, 
        bridge_params_dict[genes].end_range, 
        bridge_params_dict[genes].sigma_range, 
        bridge_params_dict[genes].upregulation_bridge)
    return simple_bridges

def generate_combined_bridges(bridge_params_dict: dict, num_genes: str, genes: str, upregulation_bridge: bool) -> np.ndarray:
        bridges_first_half = utils.generate_valid_bridges(
            bridge_params_dict[num_genes], 
            bridge_params_dict[genes].start_range, 
            bridge_params_dict[genes].start_range, 
            bridge_params_dict[genes].sigma_range, 
            bridge_params_dict[genes].upregulation_bridge)
        bridges_endpoints_first_half = [bridge[-51] for bridge in bridges_first_half]
        number_of_bridges = len(bridges_endpoints_first_half)
        bridges_second_half = []
        for i in range(number_of_bridges):
            start_range_value = bridges_endpoints_first_half[i]
            bridges_second_half = utils.generate_valid_bridges(
                bridge_params_dict[num_genes], 
                start_range=(start_range_value, start_range_value), 
                end_range=bridge_params_dict[genes].end_range, 
                sigma_range=bridge_params_dict[genes].sigma_range, 
                upregulation_bridge=upregulation_bridge
            )
            bridges_second_half.append(bridges_second_half[0])
        combined_bridges = utils.combine_bridges(bridges_first_half, bridges_second_half)
        return combined_bridges

def adjust_means_by_phase(
    phase: str,
    full_obs: Dict[str, pd.DataFrame],
    transformed_means: Dict[str, pd.DataFrame],
    marker_genes: pd.DataFrame,
    S_bridges: np.ndarray,
    G2M_bridges: np.ndarray,
    identity_bridges: np.ndarray
) -> Dict[str, pd.DataFrame]:
    """Adjust the mean expression of the genes for each cell in the given phase.
    The adjustment is done by extracting upregulation factors from bridges (create_facs_per_cell()) and multiplying them with original gene means.

    Args:
    phase (str): The current phase of the cell cycle.
    full_obs (Dict[str, pd.DataFrame]): The DataFrame containing the cells.
    transformed_means (Dict[str, pd.DataFrame]): The DataFrame containing the mean expression of the genes for each cell.
    marker_genes (DataFrame): The DataFrame containing all genes.
    S_bridges (ndarray): The bridges for the S phase.
    G2M_bridges (ndarray): The bridges for the G2M phase.
    identity_bridges (ndarray): The bridges for the identity phase.

    Returns:
    Dict[str, pd.DataFrame]: The DataFrame containing the mean expression of the genes for each cell after the adjustment.
    """
    
    phase_transf_means = {pat: [] for pat in full_obs}

    for patient in full_obs:
        df_obs = full_obs[patient].copy()
        mask_cells = (df_obs['cell_phase'] == phase).values
        new_means = transformed_means[patient].copy()
        cell_means = new_means[mask_cells]

        df_obs = df_obs[mask_cells]

        adjusted_cell_means = np.zeros_like(cell_means)
        for i in range(cell_means.shape[0]):
            common_random_index, cell_factors = create_facs_per_cell(marker_genes, S_bridges, G2M_bridges, identity_bridges)
            adjusted_cell_means[i, :] = cell_means[i, :] * cell_factors.T

            # Save the cell position for the next step: library size transformation
            cell_position_x = common_random_index / 100
            df_obs.iloc[i].cell_position_x = cell_position_x

        full_obs[patient][mask_cells] = df_obs

        new_means[mask_cells] = adjusted_cell_means
        phase_transf_means[patient] = new_means

    return phase_transf_means

def create_facs_per_cell(marker_genes: pd.DataFrame, S_bridges: np.ndarray, G2M_bridges: np.ndarray, identity_bridges: np.ndarray) -> Tuple[int, pd.DataFrame]:
    """Extract upregulation factors from bridges and assign them to the corresponding genes.
    The factors are then used to adjust the mean expression of the genes.
    
    Args:
    marker_genes (DataFrame): The DataFrame containing genes.
    S_bridges (ndarray): The bridges for the S phase.
    G2M_bridges (ndarray): The bridges for the G2M phase.
    identity_bridges (ndarray): The bridges for the identity phase.
    
    Returns:
    int: The common random index used to identify the position of the cell in the bridge/phase.
    DataFrame: The DataFrame containing the upregulation factors for each gene.
    """

    condition_s_genes = (marker_genes.S == 1)
    condition_g2m_genes = (marker_genes.G2M == 1)
    condition_identity_genes = ((marker_genes.S == 0) & (marker_genes.G2M == 0))

    mask_s_genes = (condition_s_genes).ravel()
    mask_g2m_genes = (condition_g2m_genes).ravel()
    mask_identity_genes = (condition_identity_genes).ravel()

    facs_per_cell = marker_genes[['S']].copy()

    common_random_index = np.random.randint(0, len(S_bridges[0])) # lenght of all bridges is the same = 100

    values_from_s_bridges = utils.extract_values_from_bridges(S_bridges, common_random_index)
    values_from_g2m_bridges = utils.extract_values_from_bridges(G2M_bridges, common_random_index)
    values_from_identity_bridges = utils.extract_values_from_bridges(identity_bridges, common_random_index)

    facs_per_cell['S'] = facs_per_cell['S'].astype(float)
    facs_per_cell.loc[mask_s_genes, 'S'] = values_from_s_bridges
    facs_per_cell.loc[mask_g2m_genes, 'S'] = values_from_g2m_bridges
    facs_per_cell.loc[mask_identity_genes, 'S'] = values_from_identity_bridges

    return common_random_index, facs_per_cell

######### Adjusting library size linked to phase ############
def transform_library_size(
    phase: str,
    full_obs: Dict[str, pd.DataFrame],
    transformed_means: Dict[str, pd.DataFrame],
    cell_cycle_proportions: Proportions,
    coefficient_cache: CoefficientCache
) -> Dict[str, pd.DataFrame]:
    
    phase_transf_means = {pat: [] for pat in full_obs}

    phase_transf_means = adjust_library_size_by_phase(
        phase, 
        full_obs, 
        transformed_means, 
        cell_cycle_proportions,
        coefficient_cache 
    )

    return phase_transf_means

def adjust_library_size_by_phase(
    phase: str,
    full_obs: Dict[str, pd.DataFrame],
    transformed_means: Dict[str, pd.DataFrame],
    cell_cycle_proportions: Proportions,
    coefficient_cache: CoefficientCache
) -> Dict[str, pd.DataFrame]:
    
    phase_transf_means = {pat: [] for pat in full_obs}

    for patient in full_obs:
        df_obs = full_obs[patient].copy()
        mask_cells = (df_obs['cell_phase'] == phase).values
        new_means = transformed_means[patient].copy()
        cell_means = new_means[mask_cells]

        df_obs = df_obs[mask_cells]

        for i in range(cell_means.shape[0]):
            original_proportions = cell_cycle_proportions.get_proportion(df_obs.iloc[i].program)
            adjusted_cell_position_x, G1, S, G2M = adjust_common_random_index(phase, df_obs.iloc[i].cell_position_x, original_proportions)

            df_obs.iloc[i].cell_position_x = adjusted_cell_position_x
            df_obs.iloc[i].cell_position_y = get_factor_for_new_library_size(adjusted_cell_position_x, G1, coefficient_cache)
            
        dfs_obs_cell_position_y = df_obs.cell_position_y.values
        cell_position_y_column = dfs_obs_cell_position_y.astype(float).reshape(-1, 1)
        cell_means *= cell_position_y_column

        full_obs[patient][mask_cells] = df_obs

        new_means[mask_cells] = cell_means
        phase_transf_means[patient] = new_means

    return phase_transf_means

def adjust_common_random_index(phase: str, common_random_index: int, original_proportions: dict) -> Tuple[int, Tuple[int, float], Tuple[int, float], Tuple[int, float]]:
    """
    Adjust the cell cycle proportions by excluding the G0 phase and redistributing its proportion among the other phases.
    Adjust the exact position of the cell within the phase (common_random_index) after the redistribution of the cell cycle proportions.
    
    Args:
    phase (str): The current phase of the cell cycle.
    common_random_index (int): The position of the cell within the phase.
    original_proportions (dict): The original proportions of the cell cycle phases.
    
    Returns:
    int: The adjusted position of the cell within the phase.
    Tuple[int, float]: The range of the G1 phase.
    Tuple[int, float]: The range of the S phase.
    Tuple[int, float]: The range of the G2M phase.
    """

    remaining_total = sum(value for phase, value in original_proportions.items() if phase != 'G0')
    adjusted_proportions = {phase: value / remaining_total for phase, value in original_proportions.items() if phase != 'G0'}

    G1 = (0, adjusted_proportions.get('G1'))
    S = (G1[1], G1[1] + adjusted_proportions.get('S'))
    G2M = (S[1], S[1] + adjusted_proportions.get('G2M'))

    if phase == 'G1':
        common_random_index = G1[0] + (G1[1] - G1[0]) * common_random_index
    elif phase == 'S':
        common_random_index = S[0] + (S[1] - S[0]) * common_random_index
    elif phase == 'G2M':
        common_random_index = G2M[0] + (G2M[1] - G2M[0]) * common_random_index

    return common_random_index, G1, S, G2M

def get_factor_for_new_library_size(common_random_index: int, G1: Tuple[int, float], coefficient_cache: CoefficientCache) -> float:
    G1_bottom = G1[1] / 2
    points1 = [(0, 2), (G1_bottom, 1), (G1[1], 2)]
    points2 = [(G1_bottom - (1 - G1_bottom), 2), (G1_bottom, 1), (1, 2)]

    coefficients_1, coefficients_2 = coefficient_cache.recalculate_coefficients(points1, points2)
    a, b, c = symbols('a b c')

    def quadratic_function_1(x):
        return coefficients_1[a] * x**2 + coefficients_1[b] * x + coefficients_1[c]

    def quadratic_function_2(x):
        return coefficients_2[a] * x**2 + coefficients_2[b] * x + coefficients_2[c]
    
    if common_random_index < G1_bottom:
        return quadratic_function_1(common_random_index)
    elif common_random_index > G1_bottom:
        return quadratic_function_2(common_random_index)
    else:
        return 1