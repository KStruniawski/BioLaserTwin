"""Independent high-fidelity truth generators (isolated from solvers)."""

from . import aggregate_optics, kinetics, longitudinal
from .longitudinal import (
    LongitudinalParams,
    SampleHistory,
    diffusive_refill_time_min,
    make_thermal_history_temperature_fn,
    relax_sample_history,
    update_sample_history,
)
from .aggregate_optics import (
    AggregateOpticsParams,
    compute_aggregate_scattering_cm1,
    compute_crowding_microviscosity_cp,
    compute_tht_binding,
    rgd_rod_cross_section,
    rgd_sphere_cross_section,
    trajectory_to_truth_states,
)
from .fp_cavity import FabryPerotConfig, FabryPerotTruthSimulator, ObservationBundle
from .photophysics import (
    KineticTrajectory,
    TruthSampleState,
    effective_gain_and_emission,
    simulate_secondary_nucleation_kinetics,
    spectral_reabsorption_cm1,
    wavelength_dependent_scattering,
)
from .proxy_map import LatentProxyVector, map_latents_to_truth_state

__all__ = [
    "AggregateOpticsParams",
    "FabryPerotConfig",
    "FabryPerotTruthSimulator",
    "KineticTrajectory",
    "LatentProxyVector",
    "LongitudinalParams",
    "ObservationBundle",
    "SampleHistory",
    "TruthSampleState",
    "aggregate_optics",
    "compute_aggregate_scattering_cm1",
    "compute_crowding_microviscosity_cp",
    "compute_tht_binding",
    "diffusive_refill_time_min",
    "effective_gain_and_emission",
    "kinetics",
    "longitudinal",
    "make_thermal_history_temperature_fn",
    "map_latents_to_truth_state",
    "relax_sample_history",
    "rgd_rod_cross_section",
    "rgd_sphere_cross_section",
    "simulate_secondary_nucleation_kinetics",
    "spectral_reabsorption_cm1",
    "trajectory_to_truth_states",
    "update_sample_history",
    "wavelength_dependent_scattering",
]
