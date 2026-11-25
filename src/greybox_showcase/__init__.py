"""greybox_showcase: Public mathematical showcase of a physics-informed grey-box digital twin.

Exports:
- ISO 19848:2018 / SFI Group 601 Polymorphic Twin (`AbstractReciprocatingTwin`, `ISO19848TelemetryFrame`)
- Fault-Subspace Orthogonal Projector (`FaultSubspaceProjector`, `OrthogonalPolySurrogate`)
- 8-Fault Multi-Regime Observability & Isolation Engine (`build_multi_regime_observability_matrix`,
  `evaluate_jacobian_rank_gate`, `analyze_crankshaft_roughness`, `FaultIsolationEngine`, `isolate_faults`)
"""

from greybox_showcase.iso19848_twin import (
    AbstractReciprocatingTwin,
    ISO19848ChannelMapping,
    ISO19848TelemetryFrame,
    ReciprocatingPlantConfig,
    SubModel1BreathingSnapshot,
    SubModel2ThermalSnapshot,
    SubModel3FuelTrimSnapshot,
    SubModel4IASRoughnessSnapshot,
    TwinEvaluationSnapshot,
    resolve_config_path,
)
from greybox_showcase.observability import (
    BANK_1_CYLINDERS,
    BANK_2_CYLINDERS,
    CANONICAL_FAULT_SIGNATURES,
    CYLINDER_TO_BANK_MAP,
    FAULT_PARAMETER_NAMES,
    FIRING_ORDER,
    JACOBIAN_RANK_GATE_COND_MAX,
    LEAK_SCALE_THROTTLED,
    CrankshaftRoughnessAnalysis,
    ExcitationRegime,
    FaultIsolationEngine,
    FaultModeId,
    FaultSignatureSpec,
    IsolationResult,
    JacobianRankGateResult,
    MultiRegimeObservations,
    ThermalSubFault,
    analyze_crankshaft_roughness,
    build_multi_regime_observability_matrix,
    evaluate_jacobian_rank_gate,
    finite_diff_observability_jacobian,
    isolate_faults,
    load_fault_isolation_matrix,
    numpy_multi_regime_residuals,
)
from greybox_showcase.orthogonal_projector import (
    FaultSubspaceProjector,
    OrthogonalPolySurrogate,
    SurrogateAuditResult,
)

__version__ = "0.1.1"

__all__ = [
    "__version__",
    "AbstractReciprocatingTwin",
    "ISO19848ChannelMapping",
    "ISO19848TelemetryFrame",
    "ReciprocatingPlantConfig",
    "SubModel1BreathingSnapshot",
    "SubModel2ThermalSnapshot",
    "SubModel3FuelTrimSnapshot",
    "SubModel4IASRoughnessSnapshot",
    "TwinEvaluationSnapshot",
    "resolve_config_path",
    "FaultSubspaceProjector",
    "OrthogonalPolySurrogate",
    "SurrogateAuditResult",
    "BANK_1_CYLINDERS",
    "BANK_2_CYLINDERS",
    "CANONICAL_FAULT_SIGNATURES",
    "CYLINDER_TO_BANK_MAP",
    "FAULT_PARAMETER_NAMES",
    "FIRING_ORDER",
    "JACOBIAN_RANK_GATE_COND_MAX",
    "LEAK_SCALE_THROTTLED",
    "CrankshaftRoughnessAnalysis",
    "ExcitationRegime",
    "FaultIsolationEngine",
    "FaultModeId",
    "FaultSignatureSpec",
    "IsolationResult",
    "JacobianRankGateResult",
    "MultiRegimeObservations",
    "ThermalSubFault",
    "analyze_crankshaft_roughness",
    "build_multi_regime_observability_matrix",
    "evaluate_jacobian_rank_gate",
    "finite_diff_observability_jacobian",
    "isolate_faults",
    "load_fault_isolation_matrix",
    "numpy_multi_regime_residuals",
]
