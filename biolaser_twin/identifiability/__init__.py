"""Fisher Information Matrix (FIM), structural rank check, CRB, and Bayesian DoE."""

from .doe import DesignScore, evaluate_design, select_optimal_designs
from .fisher import (
    DesignConfig,
    active_observable_mask,
    check_structural_n_l_rank,
    compute_fim,
    cramer_rao_sd,
    numerical_jacobian,
    reduced_forward_observables,
)
from .observability import (
    KineticObservableConfig,
    compute_kinetic_timeseries_fim,
    integrate_nominal_5moment_ode,
    kinetic_crb_widths,
    nominal_kinetic_forward_observables,
)
from .profile_likelihood import (
    IndistinguishablePair,
    ProfileLikelihoodResult,
    find_indistinguishable_pair,
    profile_likelihood_log10_kp,
)

__all__ = [
    "DesignConfig",
    "DesignScore",
    "IndistinguishablePair",
    "KineticObservableConfig",
    "ProfileLikelihoodResult",
    "active_observable_mask",
    "check_structural_n_l_rank",
    "compute_fim",
    "compute_kinetic_timeseries_fim",
    "cramer_rao_sd",
    "evaluate_design",
    "find_indistinguishable_pair",
    "integrate_nominal_5moment_ode",
    "kinetic_crb_widths",
    "nominal_kinetic_forward_observables",
    "numerical_jacobian",
    "profile_likelihood_log10_kp",
    "reduced_forward_observables",
    "select_optimal_designs",
]
