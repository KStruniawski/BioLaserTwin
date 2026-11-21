"""Inverse solvers (PhotonicPIELM, PlainELM, ClassicalNLS, FrameCNN, FrameViT, TabularMLP)."""

from .baselines import (
    FrameCNNSolver,
    FrameViTSolver,
    HybridCNNFeaturesSolver,
    MeanOnlySolver,
    TabularMLPSolver,
)
from .classical_nls import ClassicalNLSSolver, NLSResult
from .kinetic_classical import (
    AmyloFitGlobalNLSSolver,
    ClassicalKineticFit,
    MomentUKFSolver,
    SequentialImportanceSmoother,
)
from .kinetic_modern import (
    PILNNCfCSolver,
    PINNMambaSolver,
    PINNsFormerSolver,
    PirateNetSolver,
)
from .kinetic_pielm import (
    HierarchicalKineticPIELMSolver,
    KineticPrediction,
    KineticXTFCSolver,
    LabelsOnlyNominalODESolver,
    PlainELMTemporalSolver,
)
from .kinetic_sbi import FMPECFMSolver
from .kinetic_ude import StoichiometricUDESolver
from .photonic_pielm import PhotonicPIELMSolver, PlainELMSolver

__all__ = [
    "AmyloFitGlobalNLSSolver",
    "ClassicalKineticFit",
    "ClassicalNLSSolver",
    "FMPECFMSolver",
    "FrameCNNSolver",
    "FrameViTSolver",
    "HierarchicalKineticPIELMSolver",
    "HybridCNNFeaturesSolver",
    "KineticPrediction",
    "KineticXTFCSolver",
    "LabelsOnlyNominalODESolver",
    "MeanOnlySolver",
    "MomentUKFSolver",
    "NLSResult",
    "PILNNCfCSolver",
    "PINNMambaSolver",
    "PINNsFormerSolver",
    "PhotonicPIELMSolver",
    "PirateNetSolver",
    "PlainELMSolver",
    "PlainELMTemporalSolver",
    "SequentialImportanceSmoother",
    "StoichiometricUDESolver",
    "TabularMLPSolver",
]
