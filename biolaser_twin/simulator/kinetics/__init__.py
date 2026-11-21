"""Truth population-balance kinetics, analytical limits, and SymPy verification for BioLaserTwin v0.2."""

from .analytic import (
    cohen_2011_early_time_mass,
    elongation_only_analytical,
    oosawa_primary_only_mass,
)
from .mechanisms import MECHANISM_ZOO, MechanismSpec, effective_rate_multiplier
from .moments import (
    KineticSummaryTargets,
    compute_moment_from_bins,
    extract_kinetic_targets,
    extract_truth_kinetic_targets,
    solve_reduced_moments_length_independent,
)
from .population_balance import (
    PBETrajectory,
    build_conservative_fragmentation_matrix,
    build_log_pivot_grid,
    solve_pbe,
)

__all__ = [
    "MECHANISM_ZOO",
    "KineticSummaryTargets",
    "MechanismSpec",
    "PBETrajectory",
    "build_conservative_fragmentation_matrix",
    "build_log_pivot_grid",
    "cohen_2011_early_time_mass",
    "compute_moment_from_bins",
    "effective_rate_multiplier",
    "elongation_only_analytical",
    "extract_kinetic_targets",
    "extract_truth_kinetic_targets",
    "oosawa_primary_only_mass",
    "solve_pbe",
    "solve_reduced_moments_length_independent",
]
