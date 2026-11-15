#!/usr/bin/env python3
"""Single-command runnable showcase for greybox-ice-twin-showcase.

Demonstrates:
1. ISO 19848:2018 / SFI Group 601 Polymorphic Twin Instantiation (BMW N42B20A <-> MAN B&W 6S50ME-C)
   with zero code changes and dimensionless similarity computation.
2. Multi-Regime 8-Fault Observability Matrix:
   - Single-regime rank deficiency (rank = 4/8, cond = inf)
   - Full 4-regime stack (rank = 8/8, cond(H_tilde) = 6.413 < 50.0)
3. Fault-Subspace Orthogonal Projection P_{F^perp}:
   - Algebraic verification (idempotence, symmetry, orthogonality < 1e-12)
   - Anti-masking surrogate proof (||P_F g_perp||_2 < 1e-12)
4. Deterministic 8-Fault Disambiguation Examples:
   - FM_01 CCV air leak (sqrt(10) ~= 3.16x vacuum scaling)
   - FM_05 vs FM_06 (single-injector fouling vs Valvetronic wear)
   - FM_08 (heat-exchanger fouling)

Usage:
  python scripts/run_showcase_demo.py
  python scripts/run_showcase_demo.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from greybox_showcase import (  # noqa: E402
    AbstractReciprocatingTwin,
    FaultIsolationEngine,
    FaultSubspaceProjector,
    ISO19848TelemetryFrame,
    MultiRegimeObservations,
    OrthogonalPolySurrogate,
    evaluate_jacobian_rank_gate,
    finite_diff_observability_jacobian,
    build_multi_regime_observability_matrix,
)


def run_showcase() -> Dict[str, Any]:
    """Executes all 4 mathematical showcase pillars and returns structured metrics."""
    t0 = time.perf_counter()

    # 1. ISO 19848 Polymorphic Twin: Automotive (4-cyl 4-stroke) <-> Maritime (6-cyl 2-stroke)
    bmw_twin = AbstractReciprocatingTwin("bmw_n42b20a.yaml")
    marine_twin = AbstractReciprocatingTwin("maritime_man_bw_6s50me_c.yaml")

    bmw_frame = ISO19848TelemetryFrame(
        engine_speed_rpm=700.0,
        manifold_or_scavenge_pressure_pa=96325.0,
        charge_air_temperature_k=305.15,
        mass_air_flow_kg_s=0.0035,
        valve_lift_m=0.00035,
        coolant_temperature_k=378.15,
        oil_temperature_k=368.15,
        crankshaft_roughness_rad_s2=[0.05, -0.02, 0.04, -0.01],
    )
    marine_frame = ISO19848TelemetryFrame(
        engine_speed_rpm=127.0,
        manifold_or_scavenge_pressure_pa=320000.0,
        charge_air_temperature_k=315.15,
        mass_air_flow_kg_s=25.0,
        valve_lift_m=0.1200,
        coolant_temperature_k=358.15,
        oil_temperature_k=328.15,
        coolant_fouling_factor=0.58,
        crankshaft_roughness_rad_s2=[0.01, -0.02, 0.01, 0.02, 0.55, -0.01],
    )

    bmw_snap = bmw_twin.evaluate_step(bmw_frame)
    marine_snap = marine_twin.evaluate_step(marine_frame)
    similarity = bmw_twin.compute_dimensionless_similarity(marine_twin)

    # 2. Multi-Regime Observability Matrix & SVD Rank Gate
    single_gate = evaluate_jacobian_rank_gate(regimes=("VALVETRONIC_50MBAR",))
    full_regimes = ("KOEO", "VALVETRONIC_50MBAR", "ACTIVE_THROTTLED_500MBAR", "COLD_WARMUP")
    full_gate = evaluate_jacobian_rank_gate(regimes=full_regimes)
    H_analytical, _, _, _ = build_multi_regime_observability_matrix(regimes=full_regimes)
    J_fd = finite_diff_observability_jacobian(np.zeros(8, dtype=np.float64))
    fd_max_diff = float(np.max(np.abs(J_fd - H_analytical)))

    # 3. Orthogonal Fault-Subspace Projection P_{F^perp}
    rng = np.random.default_rng(1337)
    n_cal = 300
    rpm = rng.uniform(700.0, 3600.0, n_cal)
    lvvt_m = rng.uniform(0.0003, 0.0090, n_cal)
    m_calc = (rpm / 6000.0) * (lvvt_m / 0.0097) * 0.045 + 0.003
    dp_man = rng.uniform(4500.0, 48000.0, n_cal)

    projector = FaultSubspaceProjector().fit(m_calc, dp_man)
    proj_props = projector.verify_properties()

    r_clean = projector.project_residuals(0.0008 * np.sin(rpm / 120.0) * np.cos(lvvt_m / 0.002))
    r_fault = 0.0010 - 0.05 * m_calc - 0.0012 * np.sqrt(dp_man / 50000.0)
    r_raw = r_clean + r_fault
    X_feat = np.column_stack([rpm, lvvt_m])

    surrogate = OrthogonalPolySurrogate(degree=3, ridge_alpha=1e-3).fit(X_feat, r_raw, m_calc, dp_man)
    g_perp = surrogate.predict_surrogate(X_feat, m_calc, dp_man)
    F_basis = FaultSubspaceProjector.build_fault_basis(m_calc, dp_man)
    audit = surrogate.audit_fault_non_absorption(F_basis, g_perp)

    # 4. Deterministic 8-Fault Isolation Samples
    engine = FaultIsolationEngine()
    ccv_res = engine.isolate_faults(
        MultiRegimeObservations(
            koeo_hfm_voltage=1.001,
            koeo_maf_kg_h=0.05,
            valv_maf_residual_kg_h=-2.0,
            valv_fuel_trim_b1=0.070,
            valv_fuel_trim_b2=0.068,
            throt_maf_residual_kg_h=-6.32,
            throt_fuel_trim_b1=0.065,
            throt_fuel_trim_b2=0.063,
        )
    )
    vvt_wear_res = engine.isolate_faults(
        MultiRegimeObservations(
            koeo_hfm_voltage=1.001,
            valv_maf_residual_kg_h=-0.4,
            valv_fuel_trim_b1=0.015,
            valv_fuel_trim_b2=0.012,
            valv_rough_running=[1.65, 0.08, 0.05, 0.10],
            throt_maf_residual_kg_h=0.05,
            throt_fuel_trim_b1=0.005,
            throt_fuel_trim_b2=0.004,
            throt_rough_running=[0.12, 0.08, 0.06, 0.09],
        )
    )

    elapsed_s = round(time.perf_counter() - t0, 4)
    return {
        "project": "greybox-ice-twin-showcase",
        "elapsed_s": elapsed_s,
        "iso19848_polymorphic_twin": {
            "automotive": {
                "plant_id": bmw_twin.plant_id,
                "cylinders": bmw_twin.cylinders,
                "strokes_per_cycle": bmw_twin.strokes_per_cycle,
                "m_dot_calc_kg_s": round(bmw_snap.submodel1_breathing.m_dot_calc_kg_s, 6),
                "gate4_passed": bmw_snap.submodel4_ias_roughness.gate4_passed,
            },
            "maritime": {
                "plant_id": marine_twin.plant_id,
                "cylinders": marine_twin.cylinders,
                "strokes_per_cycle": marine_twin.strokes_per_cycle,
                "m_dot_calc_kg_s": round(marine_snap.submodel1_breathing.m_dot_calc_kg_s, 4),
                "fouling_factor_coolant": marine_snap.submodel2_thermal.fouling_factor_coolant,
                "fouling_alert": marine_snap.submodel2_thermal.fouling_alert,
                "ias_peak_cylinder": marine_snap.submodel4_ias_roughness.peak_cylinder,
            },
            "similarity": {k: round(v, 4) for k, v in similarity.items()},
        },
        "multi_regime_observability": {
            "single_regime_rank": single_gate.rank,
            "single_regime_cond": "inf" if np.isinf(single_gate.condition_number) else round(single_gate.condition_number, 3),
            "full_stack_rank": full_gate.rank,
            "full_stack_cond_h_tilde": round(full_gate.condition_number, 4),
            "jacobian_gate_passed": full_gate.passed,
            "finite_diff_max_error": fd_max_diff,
        },
        "orthogonal_projector": {
            "idempotence_error": proj_props["idempotence_error"],
            "symmetry_error": proj_props["symmetry_error"],
            "orthogonality_error": proj_props["orthogonality_error"],
            "surrogate_fault_subspace_norm": audit.fault_subspace_norm,
            "anti_masking_verified": audit.is_anti_masking_verified,
        },
        "fault_isolation_samples": {
            "ccv_leak": {
                "isolated_fault": ccv_res.primary_fault.value,
                "sfi_code": ccv_res.sfi_code,
                "leak_ratio_500_to_50": round(ccv_res.evidence.get("leak_ratio_500_to_50", 0.0), 3),
            },
            "valvetronic_wear": {
                "isolated_fault": vvt_wear_res.primary_fault.value,
                "sub_fault": vvt_wear_res.sub_fault,
                "sfi_code": vvt_wear_res.sfi_code,
            },
        },
    }


def print_banner(payload: Dict[str, Any]) -> None:
    """Prints a concise terminal summary of the showcase results."""
    iso = payload["iso19848_polymorphic_twin"]
    obs = payload["multi_regime_observability"]
    proj = payload["orthogonal_projector"]
    faults = payload["fault_isolation_samples"]

    sep = "=" * 84
    print(sep)
    print("  GREYBOX-ICE-TWIN-SHOWCASE | Physics-Informed Digital Twin Mathematical Core")
    print(sep)
    print("\n[1] ISO 19848:2018 / SFI 601 Polymorphic Twin (Zero Code Branching)")
    print(f"  - Automotive : {iso['automotive']['plant_id']} ({iso['automotive']['cylinders']}-cyl, {iso['automotive']['strokes_per_cycle']}-stroke) -> m_dot = {iso['automotive']['m_dot_calc_kg_s']} kg/s")
    print(f"  - Maritime   : {iso['maritime']['plant_id']} ({iso['maritime']['cylinders']}-cyl, {iso['maritime']['strokes_per_cycle']}-stroke) -> m_dot = {iso['maritime']['m_dot_calc_kg_s']} kg/s, fouling={iso['maritime']['fouling_factor_coolant']} (alert={iso['maritime']['fouling_alert']})")
    print(f"  - Similarity : Mach discrepancy = {iso['similarity']['mach_critical_discrepancy']}, Strouhal ratio = {iso['similarity']['strouhal_scale_ratio']}, score = {iso['similarity']['isomorphic_transferability_score']}")

    print("\n[2] Multi-Regime 8-Fault Observability & SVD Rank Gate")
    print(f"  - Single Regime (Valvetronic 50 mbar) : rank = {obs['single_regime_rank']}/8, cond = {obs['single_regime_cond']}")
    print(f"  - Stacked 4-Regime Excitation         : rank = {obs['full_stack_rank']}/8, cond(H_tilde) = {obs['full_stack_cond_h_tilde']} (< 50.0) | PASS={obs['jacobian_gate_passed']}")
    print(f"  - Finite-Difference Jacobian Parity   : max |J_fd - H| = {obs['finite_diff_max_error']:.2e}")

    print("\n[3] Fault-Subspace Orthogonal Projection P_{F^perp}")
    print(f"  - Idempotence ||P^2 - P||_F           : {proj['idempotence_error']:.2e}")
    print(f"  - Orthogonality ||P F||_F             : {proj['orthogonality_error']:.2e}")
    print(f"  - Anti-Masking ||P_F g_perp||_2       : {proj['surrogate_fault_subspace_norm']:.2e} (verified={proj['anti_masking_verified']})")

    print("\n[4] Deterministic Fault Isolation Samples")
    print(f"  - CCV Vacuum Leak (50 -> 500 mbar)    : {faults['ccv_leak']['isolated_fault']} (scale={faults['ccv_leak']['leak_ratio_500_to_50']}x, SFI {faults['ccv_leak']['sfi_code']})")
    print(f"  - Valvetronic Wear (0.35 -> 9.7 mm)   : {faults['valvetronic_wear']['isolated_fault']} ({faults['valvetronic_wear']['sub_fault']}, SFI {faults['valvetronic_wear']['sfi_code']})")

    print(f"\nCompleted in {payload['elapsed_s']:.3f} s.")
    print(sep)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run greybox-ice-twin-showcase demo.")
    parser.add_argument("--json", action="store_true", help="Output structured JSON to stdout.")
    args = parser.parse_args(argv)

    payload = run_showcase()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_banner(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
