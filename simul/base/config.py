import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple, Dict, Union, List
from sympy import symbols, Eq, solve

class Proportions:
    DEFAULT_PROPORTION = [0.10, 0.54, 0.20, 0.16]
    PHASES = ["G0", "G1", "S", "G2M"]

    def __init__(self, celltypes, proportions):
        self.proportions = {}
        for i, cell_type in enumerate(celltypes):
            if i < len(proportions):
                if self._validate_proportions(proportions[i]):
                    self.proportions[cell_type] = dict(zip(self.PHASES, proportions[i]))
                else:
                    raise ValueError(f"Proportions for {cell_type} do not sum up to 1.")
            else:
                self.proportions[cell_type] = dict(zip(self.PHASES, self.DEFAULT_PROPORTION))

    @staticmethod
    def _validate_proportions(proportion_list):
        return abs(sum(proportion_list) - 1) < 1e-6

    def get_proportion(self, cell_type):
        return self.proportions.get(cell_type, {})
    
class CoefficientCache:
    """Cache for the coefficients of the quadratic equations used to calculate the function for adjustment of library size."""
    def __init__(self):
        self.cache = {}

    def recalculate_coefficients(self, points1, points2):
        a, b, c = symbols('a b c')
        key = (tuple(points1), tuple(points2))

        if key in self.cache:
            return self.cache[key]

        eq1 = Eq(points1[0][1], a * points1[0][0]**2 + b * points1[0][0] + c)
        eq2 = Eq(points1[1][1], a * points1[1][0]**2 + b * points1[1][0] + c)
        eq3 = Eq(points1[2][1], a * points1[2][0]**2 + b * points1[2][0] + c)
        solutions_1 = solve((eq1, eq2, eq3), (a, b, c))

        eq4 = Eq(points2[0][1], a * points2[0][0]**2 + b * points2[0][0] + c)
        eq5 = Eq(points2[1][1], a * points2[1][0]**2 + b * points2[1][0] + c)
        eq6 = Eq(points2[2][1], a * points2[2][0]**2 + b * points2[2][0] + c)
        solutions_2 = solve((eq4, eq5, eq6), (a, b, c))

        self.cache[key] = (solutions_1, solutions_2)
        return solutions_1, solutions_2

@dataclass
class BridgeParam:
    start_range: Tuple[float, float]
    end_range: Tuple[float, float]
    sigma_range: Tuple[float, float]
    upregulation_bridge: bool

@dataclass
class Bridges:
    G1_phase_S_genes: BridgeParam
    S_phase_S_genes: BridgeParam
    G2M_phase_S_genes: BridgeParam

    G1_phase_G2M_genes: BridgeParam
    S_phase_G2M_genes: BridgeParam
    G2M_phase_G2M_genes: BridgeParam

    G1_phase_identity_genes: BridgeParam
    S_phase_identity_genes: BridgeParam
    G2M_phase_identity_genes: BridgeParam
    
    @staticmethod
    def default() -> 'Bridges':
        return Bridges(
            # Path for S marker genes
            G1_phase_S_genes=BridgeParam(start_range=(0.9, 1.1), end_range=(0.9, 1.3), sigma_range=(0.8, 0.8), upregulation_bridge=True),
            S_phase_S_genes=BridgeParam(start_range=(1, 1.5), end_range=(1, 1.5), sigma_range=(0.8, 0.8), upregulation_bridge=True),
            G2M_phase_S_genes=BridgeParam(start_range=(1, 1.5), end_range=(0.9, 1.1), sigma_range=(0.8, 0.8), upregulation_bridge=False),

            # Path for G2M marker genes
            G1_phase_G2M_genes=BridgeParam(start_range=(1, 1.5), end_range=(0.9, 1.1), sigma_range=(0.8, 0.8), upregulation_bridge=False),
            S_phase_G2M_genes=BridgeParam(start_range=(0.9, 1.1), end_range=(0.9, 1.3), sigma_range=(0.8, 0.8), upregulation_bridge=True),
            G2M_phase_G2M_genes=BridgeParam(start_range=(1, 1.5), end_range=(1, 1.5), sigma_range=(0.8, 0.8), upregulation_bridge=True),
            
            # Path for identity genes
            G1_phase_identity_genes=BridgeParam(start_range=(0.9, 1.1), end_range=(0.8, 0.9), sigma_range=(0.8, 0.8), upregulation_bridge=False),
            S_phase_identity_genes=BridgeParam(start_range=(0.7, 0.9), end_range=(0.7, 0.9), sigma_range=(0.8, 0.8), upregulation_bridge=False),
            G2M_phase_identity_genes=BridgeParam(start_range=(0.8, 0.9), end_range=(0.9, 1.1), sigma_range=(0.8, 0.8), upregulation_bridge=True)
        )

class SimCellConfig:
    def __init__(
        self,
        random_seed: int = 0,
        n_genes: int = 5000,
        n_cells: int = 200,
        group_names: Union[np.ndarray, List[str]] = ["group1"],
        p_de_list: Union[float, np.ndarray] = 0.1,
        p_down_list: Union[float, np.ndarray] = 0.5,
        de_location_list: Union[float, np.ndarray] = 0.1,
        de_scale_list: Union[float, np.ndarray] = 0.4,
        batch_effect: bool = True,
        batch_names: Union[np.ndarray, List[str]] = ["patient1"],
        pb_de_list: Union[float, np.ndarray] = 0.1,
        pb_down_list: Union[float, np.ndarray] = 0.5,
        bde_location_list: Union[float, np.ndarray] = 0.1,
        bde_scale_list: Union[float, np.ndarray] = 0.4,
        shared_cnv: bool = False,
        mean_shape: float = 0.6,
        mean_scale: float = 3,
        p_outlier: float = 0.05,
        outlier_loc: int = 4,
        outlier_scale: float = 0.5,
        libsize_loc: int = 11,
        libsize_scale: float = 0.2,
        common_disp: float = 0.1,
        dof: int = 60,
        dropout_midpoint: float = 0,
        dropout_shape: float = -1,
        cell_cycle_proportions: Proportions = None,
        bridge_params: Bridges = None,
        transform_means_cell_cycle: bool = True,
        adjust_lib_size_cell_cycle: bool = True,
    ):
        self.random_seed = random_seed
        self.n_genes = n_genes
        self.n_cells = n_cells
        self.group_names = group_names
        self.p_de_list = p_de_list
        self.p_down_list = p_down_list
        self.de_location_list = de_location_list
        self.de_scale_list = de_scale_list
        self.batch_names = batch_names
        self.batch_effect = batch_effect
        self.pb_de_list = pb_de_list
        self.pb_down_list = pb_down_list
        self.bde_location_list = bde_location_list
        self.bde_scale_list = bde_scale_list
        self.shared_cnv = shared_cnv
        self.mean_shape = mean_scale
        self.mean_scale = mean_shape
        self.p_outlier = p_outlier
        self.outlier_loc = outlier_loc
        self.outlier_scale = outlier_scale
        self.libsize_loc = libsize_loc
        self.libsize_scale = libsize_scale
        self.common_disp = common_disp
        self.dof = dof
        self.dropout_midpoint = dropout_midpoint
        self.dropout_shape = dropout_shape
        self.cell_cycle_proportions = cell_cycle_proportions
        self.bridge_params = bridge_params if bridge_params is not None else Bridges.default()
        self.transform_means_cell_cycle = transform_means_cell_cycle
        self.adjust_lib_size_cell_cycle = adjust_lib_size_cell_cycle

    def to_dict(self):
        dict = vars(self).copy()
        for k, v in dict.items():
            if type(v) == np.ndarray:
                dict[k] = dict[k].tolist()
            elif isinstance(v, Proportions):
                dict[k] = v.proportions  # Convert Proportions to its dictionary representation
        return dict

    def create_rng(self):
        return np.random.default_rng(self.random_seed)
