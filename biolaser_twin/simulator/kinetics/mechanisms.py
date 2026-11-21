"""Kinetic mechanism specifications for the amyloid aggregation zoo (M0..M5, M_w)."""

from dataclasses import dataclass
import math
from typing import Callable


@dataclass(frozen=True)
class MechanismSpec:
    """Specification of an amyloid aggregation mechanism in the BioLaserTwin v0.2 zoo.

    Calibrated to realistic alpha-synuclein timescales (t50 ~ 150-450 min, fibril lengths
    crossing the RGD k*L ~ 1 boundary at ell ~ 120 monomers and reaching 300-3000 monomers).
    Concentrations are in uM and time in minutes.
    """

    name: str
    description: str
    k_n: float = 1e-7  # uM^(1 - n_c) min^-1
    n_c: float = 2.0  # primary nucleation reaction order
    k_plus: float = 0.05  # uM^-1 min^-1 (microscopic elongation rate)
    k_2: float = 0.0  # uM^(-n_2) min^-1 (secondary nucleation rate)
    n_2: float = 2.0  # secondary nucleation reaction order
    k_m_sat: float = float("inf")  # uM (Michaelis-like secondary saturation K_M)
    k_f: float = 0.0  # min^-1 (fragmentation rate per internal bond)
    k_c: float = 10.0  # min^-1 (oligomer-to-fibril conversion rate)
    k_d: float = 0.0  # min^-1 (on-pathway oligomer dissociation rate)
    k_off: float = 0.0  # min^-1 (off-pathway oligomer trapping rate)
    k_d_off: float = 0.0  # min^-1 (off-pathway oligomer dissociation rate)
    q_oligomer: int = 4  # monomer stoichiometry of an oligomer / minimal fibril
    c_total_um: float = 50.0  # total monomer-equivalent concentration
    gamma_tht_per_mm: float = 0.0  # M_w: dye-modulated rate coupling (1 + gamma * [ThT]_mM)
    ea_over_r_k: float = 0.0  # M_w: Arrhenius activation energy E_a / R in Kelvin
    t_ref_k: float = 298.15  # reference temperature in Kelvin


MECHANISM_ZOO: dict[str, MechanismSpec] = {
    "M0": MechanismSpec(
        name="M0",
        description="Primary nucleation + elongation only (Oosawa limit).",
        k_n=1.2e-7,
        n_c=2.0,
        k_plus=0.06,
        k_2=0.0,
        k_f=0.0,
        k_c=10.0,
        k_d=0.0,
    ),
    "M1": MechanismSpec(
        name="M1",
        description="Primary nucleation + elongation + conservative fragmentation.",
        k_n=2.0e-8,
        n_c=2.0,
        k_plus=0.05,
        k_2=0.0,
        k_f=1.2e-6,
        k_c=10.0,
        k_d=0.0,
    ),
    "M2": MechanismSpec(
        name="M2",
        description="Primary + unsaturated secondary nucleation + elongation.",
        k_n=2.0e-9,
        n_c=2.0,
        k_plus=0.05,
        k_2=8.0e-8,
        n_2=2.0,
        k_m_sat=float("inf"),
        k_f=0.0,
        k_c=0.5,
        k_d=0.01,
    ),
    "M3": MechanismSpec(
        name="M3",
        description="Primary + saturated secondary nucleation (K_M) + elongation.",
        k_n=2.0e-9,
        n_c=2.0,
        k_plus=0.05,
        k_2=2.0e-7,
        n_2=2.0,
        k_m_sat=20.0,
        k_f=0.0,
        k_c=0.5,
        k_d=0.01,
    ),
    "M4": MechanismSpec(
        name="M4",
        description="Oligomer-conversion-limited secondary nucleation (slow k_c, k_d).",
        k_n=5.0e-8,
        n_c=2.0,
        k_plus=0.04,
        k_2=2.5e-7,
        n_2=2.0,
        k_m_sat=25.0,
        k_f=0.0,
        k_c=0.008,
        k_d=0.002,
    ),
    "M5": MechanismSpec(
        name="M5",
        description="Secondary nucleation with off-pathway oligomer sink (k_off).",
        k_n=1.5e-6,
        n_c=2.0,
        k_plus=0.035,
        k_2=5.0e-6,
        n_2=2.0,
        k_m_sat=25.0,
        k_f=0.0,
        k_c=0.001,
        k_d=0.0005,
        k_off=0.08,
        k_d_off=5.0e-5,
    ),
    "M_w": MechanismSpec(
        name="M_w",
        description="Withheld validation mechanism: structural reaction-order shift + ThT/Arrhenius coupling.",
        k_n=8.0e-11,
        n_c=3.0,
        k_plus=0.045,
        k_2=3.0e-5,
        n_2=1.0,
        k_m_sat=15.0,
        k_f=8.0e-7,
        k_c=0.03,
        k_d=0.003,
        gamma_tht_per_mm=0.04,
        ea_over_r_k=4500.0,
        t_ref_k=298.15,
    ),
}


def effective_rate_multiplier(
    spec: MechanismSpec,
    tht_conc_mm: float = 0.0,
    temperature_k: float | Callable[[float], float] = 298.15,
    t_min: float = 0.0,
) -> float:
    """Compute dimensionless rate multiplier from ThT modulation and Arrhenius heating."""
    temp_val = float(temperature_k(t_min)) if callable(temperature_k) else float(temperature_k)
    dye_factor = max(0.1, 1.0 + spec.gamma_tht_per_mm * max(0.0, tht_conc_mm))
    if spec.ea_over_r_k == 0.0:
        arrhenius = 1.0
    else:
        arrhenius = math.exp(-spec.ea_over_r_k * (1.0 / max(temp_val, 200.0) - 1.0 / spec.t_ref_k))
    return dye_factor * arrhenius
