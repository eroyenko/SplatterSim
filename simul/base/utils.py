import pandas as pd
import anndata as ad
import scanpy as sc
import scipy
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import itertools

from typing import Optional, List, Dict, Tuple
from matplotlib.colors import TwoSlopeNorm

from scipy.stats import norm
from scipy.interpolate import CubicSpline

from ..patients.dataset import Dataset


####### Plotting ###########
def plot_subclone_profile(dataset: Dataset, filename: Optional[str] = None) -> None:
    """Function to plot the true CNV profile as a heatmap

    Args:

        dataset: an instantiated dataset object
        filename: if not None, will save the figure in the provided path

    """
    subclone_df = dataset.get_subclone_profiles()
    subclone_plot_df = dataset.order_subclone_profile(subclone_df=subclone_df)
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    sns.heatmap(subclone_plot_df, center=0, cmap="vlag", ax=ax)

    if filename is not None:
        fig.savefig(filename, bbox_inches="tight")


def plot_cnv_heatmap(
    dataset: Dataset,
    patient: str,
    var: pd.DataFrame,
    figsize: Tuple[int, int] = (15, 10),
    filename: Optional[str] = None,
) -> None:

    subclone_df = dataset.get_subclone_profiles()
    subclone_plot_df = dataset.order_subclone_profile(subclone_df=subclone_df)

    pat_subclones = subclone_plot_df.loc[subclone_plot_df.index.str.endswith(patient)]

    cnv_adata = ad.AnnData(pat_subclones.reset_index(drop=True), obs=pat_subclones.reset_index()[["index"]])

    cnv_adata.obs.columns = ["subclone"]

    chromosomes = [f"chr{i}" for i in np.append(np.arange(1, 23), ["X", "Y", "M"])]
    chr_pos_dict = {}
    for chrom in chromosomes:
        chr_pos_dict[chrom] = np.where(var.loc[subclone_plot_df.columns].chromosome == chrom)[0][0]

    chr_pos = list(chr_pos_dict.values())

    # center color map at 0
    tmp_data = cnv_adata.X.data if scipy.sparse.issparse(cnv_adata.X) else cnv_adata.X
    norm = TwoSlopeNorm(0, vmin=np.nanmin(tmp_data), vmax=np.nanmax(tmp_data))

    # add chromosome annotations
    var_group_positions = list(zip(chr_pos, chr_pos[1:] + [cnv_adata.shape[1]]))

    return_ax_dic = sc.pl.heatmap(
        cnv_adata,
        var_names=cnv_adata.var.index.values,
        groupby="subclone",
        figsize=figsize,
        cmap="vlag",
        show_gene_labels=False,
        var_group_positions=var_group_positions,
        var_group_labels=list(chr_pos_dict.keys()),
        norm=norm,
        show=False,
    )

    return_ax_dic["heatmap_ax"].vlines(chr_pos[1:], ymin=-1, ymax=cnv_adata.shape[0])

    if filename is not None:
        return_ax_dic["heatmap_ax"].figure.savefig(filename, bbox_inches="tight")

def plot_cell_cycle(df, program_name):
    """Plots the cell cycle distribution for a given program.

    Parameters:
    df (DataFrame): A DataFrame containing the data.
    program_name (str): The program to filter on.
    """

    program_df = df[df['program'] == program_name]
    phase_order = ['G0', 'G1', 'S', 'G2M']
    phase_counts = program_df['cell_phase'].value_counts().reindex(phase_order)

    phase_color_map = {
        'G0': '#1f77b4',  # Muted blue
        'G1': '#ff7f0e',  # Safety orange
        'S': '#2ca02c',   # Cooked asparagus green
        'G2M': '#d62728'  # Brick red
    }
    colors = [phase_color_map[phase] for phase in phase_counts.index]
    plot_labels = [phase.replace('G2M', 'G2/M') for phase in phase_counts.index]

    fig, ax = plt.subplots()
    ax.pie(phase_counts, labels=plot_labels, startangle=90, counterclock=False, wedgeprops=dict(width=0.3), colors=colors)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)
    ax.axis('equal')  
    plt.title(f'Cell Distribution for Program: {program_name}')
    plt.show()

######### Prob distributions ############
def generate_anchor_alphas(
    anchors: List[str],
    start_alpha: List[int] = [5, 5, 5],
    alpha_add: int = 10,
) -> Dict[Tuple, List[int]]:
    """Function to generate the alphas for the dirichlet distribution associated with each anchor
    combination (2^n_anchors)

    Args:

        anchors: the list of anchors
        start_alpha: optionally, an initial alpha distribution to start from. Eg, if you want to create a "rare"
            program, you can start off with (10, 10, 5). Defaults to (5, 5, 5)
        alpha_add: optionally, the alpha to add when the anchor is gained. Defaults to 10.

    Returns:

        a dictionary with the anchor combination as key
        (eg (True, False, True) if anchor 1 and 3 are gained)
        and the associated alphas as value

    """
    l = [False, True]
    anchor_profiles = list(itertools.product(l, repeat=len(anchors)))
    alphas = {}
    for profile in anchor_profiles:
        alphas[profile] = [x + alpha_add if profile[i] else x for i, x in enumerate(start_alpha)]
    return alphas

######### Brownian Bridge ############
def bridge(x=0, y=0, N=5, n=100, sigma_fac=0.8, lower_bound=1, upper_bound=1.5):
    """Function to generate a Brownian bridge between two points x and y.
    Adapted from Splatter's R implementation.

    Args:

            x: the start point
            y: the end point
            N: the number of steps
            n: the number of points to interpolate
            sigma_fac: the sigma factor
            lower_bound: the lower bound for the bridge
            upper_bound: the upper bound for the bridge

    Returns:

            a Brownian bridge
    """
    dt = 1 / (N - 1)
    t = np.linspace(0, 1, N)
    sigma2 = np.random.uniform(0, sigma_fac * np.mean([x, y]))
    X = np.concatenate(([0], np.cumsum(norm.rvs(scale=sigma2, size=N - 1) * np.sqrt(dt))))
    BB = x + X - t * (X[-1] - y + x)

    # Using CubicSpline with 'natural' boundary conditions
    cs = CubicSpline(t, BB, bc_type='natural')
    BB_smooth = cs(np.linspace(0, 1, n))
    BB_smooth = np.clip(BB_smooth, lower_bound, upper_bound)

    return BB_smooth

def combine_bridges(bridges_first_half: List[np.ndarray], bridges_second_half: List[np.ndarray]) -> List[np.ndarray]:
    """Function to combine two bridges into a single bridge by concatenating the first 50 values of each bridge.

    Args:

        bridges_first_half: the first set of bridges
        bridges_second_half: the second set of bridges

    Returns:

            a new combined bridge
    """
    combined_bridges = []
    for bridge_first, bridge_second in zip(bridges_first_half, bridges_second_half):
        if len(bridge_first) >= 50 and len(bridge_second) >= 50:
            combined_bridge = np.concatenate([bridge_first[:50], bridge_second[:50]])
            combined_bridges.append(combined_bridge)
    return combined_bridges

def check_bridge_form(bridge: List[float], upregulation_bridge: bool=True) -> bool:
    """Function to check if a bridge has the correct form - no multiple peaks and dips.

    Args:
    
            bridge: the bridge to check
            upregulation_bridge: whether the bridge should be upregulated or downregulated

    Returns:
    
                a boolean indicating whether the bridge has the correct form
    """
    descending = False
    for i in range(1, len(bridge)):
        if upregulation_bridge:
            if bridge[i] <= bridge[i-1]:
                descending = True
            elif descending and bridge[i] > bridge[i-1]:
                return False
        else:
            if bridge[i] >= bridge[i-1]:
                descending = True
            elif descending and bridge[i] < bridge[i-1]:
                return False
    return True

def generate_valid_bridges(num_bridges: int, start_range: Tuple[float, float], end_range: Tuple[float, float], sigma_range: Tuple[float, float], upregulation_bridge: bool):
    """Function to generate a set of valid bridges.

    Args:

        num_bridges: the number of bridges to generate
        start_range: the range of the start point
        end_range: the range of the end point
        sigma_range: the range of the sigma factor
        upregulation_bridge: whether the bridge should be upregulated or downregulated

    Returns:

            a set of valid bridges
    """
    valid_bridges = []
    lower_bound = min(start_range[0], end_range[0])
    upper_bound = max(start_range[1], end_range[1])

    while len(valid_bridges) != num_bridges:
        batch_size = num_bridges - len(valid_bridges)
        
        sp = np.random.uniform(*start_range, size=batch_size)
        ep = np.random.uniform(*end_range, size=batch_size)
        sf = np.random.uniform(*sigma_range, size=batch_size)
        
        bridges = [bridge(x=sp[i], y=ep[i], N=5, n=100, sigma_fac=sf[i], lower_bound=lower_bound, upper_bound=upper_bound) for i in range(batch_size)]

        valid_mask = np.array([check_bridge_form(b, upregulation_bridge=upregulation_bridge) for b in bridges])
        valid_bridges.extend(np.array(bridges)[valid_mask])

    return valid_bridges

def extract_values_from_bridges(bridges: List[np.ndarray], common_random_index: int):
    values_from_bridges = np.array([bridge[common_random_index] for bridge in bridges])
    return values_from_bridges