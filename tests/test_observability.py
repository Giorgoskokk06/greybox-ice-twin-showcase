"""Unit tests for 8-fault multi-regime observability matrix and isolation engine."""

from __future__ import annotations

import numpy as np
import pytest

from greybox_showcase.observability import (
    BANK_1_CYLINDERS,
    BANK_2_CYLINDERS,
    CYLINDER_TO_BANK_MAP,
    FIRING_ORDER,
    GATE4_MAX_ROUGHNESS_RAD_S2,
    JACOBIAN_RANK_GATE_COND_MAX,
    FaultIsolationEngine,
    FaultModeId,
    MultiRegimeObservations,
    ThermalSubFault,
    analyze_crankshaft_roughness,
    build_multi_regime_observability_matrix,
    evaluate_jacobian_rank_gate,
    finite_diff_observability_jacobian,
)


class TestJacobianObservabilityAndRankGate:
    """Verifies Jacobian conditioning, multi-regime rank, and finite-diff parity."""

    def test_full_stacked_regimes_rank_and_conditioning(self) -> None:
        regimes = ("KOEO", "VALVETRONIC_50MBAR", "ACTIVE_THROTTLED_500MBAR", "COLD_WARMUP")
        H, H_tilde, _, params = build_multi_regime_observability_matrix(regimes=regimes)

        assert H.shape == (12, 8)
        assert H_tilde.shape == (12, 8)
        assert len(params) == 8

        gate = evaluate_jacobian_rank_gate(regimes=regimes)
        assert gate.passed is True
        assert gate.rank == 8
        assert gate.requires_active_excitation is False
        assert gate.recommended_excitation is None
        assert gate.condition_number < JACOBIAN_RANK_GATE_COND_MAX
        assert abs(gate.condition_number - 6.413) < 0.15

    def test_single_valvetronic_idle_regime_rank_deficient(self) -> None:
        regimes = ("VALVETRONIC_50MBAR",)
        H, H_tilde, _, _ = build_multi_regime_observability_matrix(regimes=regimes)
        assert H.shape[0] == 4
        assert int(np.linalg.matrix_rank(H_tilde)) == 4

        gate = evaluate_jacobian_rank_gate(regimes=regimes)
        assert gate.passed is False
        assert gate.requires_active_excitation is True
        assert gate.condition_number >= JACOBIAN_RANK_GATE_COND_MAX

    def test_missing_koeo_and_throttled_ill_conditioned(self) -> None:
        gate = evaluate_jacobian_rank_gate(regimes=("VALVETRONIC_50MBAR", "COLD_WARMUP"))
        assert gate.passed is False
        assert gate.requires_active_excitation is True
        assert gate.condition_number > 100.0

    def test_finite_difference_jacobian_parity(self) -> None:
        H_analytical, _, _, _ = build_multi_regime_observability_matrix()
        J_fd = finite_diff_observability_jacobian(np.zeros(8, dtype=np.float64))
        assert J_fd.shape == H_analytical.shape
        assert float(np.max(np.abs(J_fd - H_analytical))) < 1e-10


class TestCrankshaftRoughnessAndHarmonics:
    """Verifies crankshaft roughness Gate 4 and harmonic order decomposition."""

    def test_firing_order_and_bank_topology(self) -> None:
        assert FIRING_ORDER == (1, 3, 4, 2)
        assert BANK_1_CYLINDERS == (1, 4)
        assert BANK_2_CYLINDERS == (2, 3)
        assert CYLINDER_TO_BANK_MAP == {1: 1, 2: 2, 3: 2, 4: 1}

    def test_order_1_maps_to_bank_differential(self) -> None:
        rr = [1.2, 0.2, 0.4, 0.8]
        analysis = analyze_crankshaft_roughness(rr)
        expected_bank_diff = 0.5 * abs((rr[0] + rr[3]) - (rr[1] + rr[2]))
        assert abs(analysis.order_1_0_mag - expected_bank_diff) < 1e-12
        assert analysis.max_spread_rad_s2 >= GATE4_MAX_ROUGHNESS_RAD_S2
        assert analysis.gate4_passed is False


class TestFaultIsolationEngine:
    """Tests deterministic disambiguation across all 8 canonical fault modes."""

    @pytest.fixture
    def engine(self) -> FaultIsolationEngine:
        return FaultIsolationEngine()

    def test_healthy_baseline_isolation(self, engine: FaultIsolationEngine) -> None:
        obs = MultiRegimeObservations(
            koeo_hfm_voltage=1.002,
            koeo_maf_kg_h=0.08,
            valv_maf_residual_kg_h=-0.15,
            valv_fuel_trim_b1=0.008,
            valv_fuel_trim_b2=0.006,
            valv_rough_running=[0.05, -0.04, 0.06, -0.03],
            throt_maf_residual_kg_h=-0.20,
            throt_fuel_trim_b1=0.005,
            throt_fuel_trim_b2=0.007,
            throt_rough_running=[0.04, -0.05, 0.05, -0.04],
            vanos_tracking_error_deg=0.25,
            vanos_slew_rate_deg_s=95.0,
            t_oil_c=92.0,
            t_coolant_c=91.0,
            t_rad_out_c=45.0,
            pwm_stat=0.0,
            theta_rad_health=0.98,
        )
        res = engine.isolate_faults(obs)
        assert res.primary_fault == FaultModeId.HEALTHY_BASELINE
        assert res.requires_active_excitation is False

    def test_all_8_fault_modes(self, engine: FaultIsolationEngine) -> None:
        # FM_01: CCV leak (3.16x scaling)
        res1 = engine.isolate_faults(
            MultiRegimeObservations(
                koeo_hfm_voltage=1.001,
                koeo_maf_kg_h=0.05,
                valv_maf_residual_kg_h=-2.0,
                valv_fuel_trim_b1=0.07,
                valv_fuel_trim_b2=0.068,
                throt_maf_residual_kg_h=-6.32,
                throt_fuel_trim_b1=0.065,
                throt_fuel_trim_b2=0.063,
            )
        )
        assert res1.primary_fault == FaultModeId.FM_01_UNMETERED_AIR_CCV_LEAK

        # FM_02: MAF gain drift (invariant to vacuum switch)
        res2 = engine.isolate_faults(
            MultiRegimeObservations(
                koeo_hfm_voltage=1.002,
                koeo_maf_kg_h=0.02,
                valv_maf_residual_kg_h=-2.5,
                valv_fuel_trim_b1=0.085,
                valv_fuel_trim_b2=0.082,
                throt_maf_residual_kg_h=-2.6,
                throt_fuel_trim_b1=0.084,
                throt_fuel_trim_b2=0.083,
            )
        )
        assert res2.primary_fault == FaultModeId.FM_02_HFM5_MAF_GAIN_DRIFT

        # FM_03: MAF zero-offset drift (KOEO gate)
        res3 = engine.isolate_faults(MultiRegimeObservations(koeo_hfm_voltage=1.085, koeo_maf_kg_h=1.2))
        assert res3.primary_fault == FaultModeId.FM_03_HFM5_MAF_ZERO_OFFSET

        # FM_04: FPR drop
        res4 = engine.isolate_faults(
            MultiRegimeObservations(
                koeo_hfm_voltage=1.000,
                valv_maf_residual_kg_h=0.10,
                valv_fuel_trim_b1=0.06,
                valv_fuel_trim_b2=0.058,
                cruise_fuel_trim_b1=0.145,
                cruise_fuel_trim_b2=0.142,
            )
        )
        assert res4.primary_fault == FaultModeId.FM_04_FUEL_PRESSURE_REGULATOR_DROP

        # FM_05: Single-injector fouling (persists at 9.7 mm)
        res5 = engine.isolate_faults(
            MultiRegimeObservations(
                koeo_hfm_voltage=1.002,
                valv_maf_residual_kg_h=0.1,
                valv_fuel_trim_b1=0.075,
                valv_fuel_trim_b2=0.010,
                valv_rough_running=[1.90, 0.05, 0.04, 0.08],
                throt_maf_residual_kg_h=0.1,
                throt_fuel_trim_b1=0.072,
                throt_fuel_trim_b2=0.012,
                throt_rough_running=[1.85, 0.04, 0.05, 0.06],
            )
        )
        assert res5.primary_fault == FaultModeId.FM_05_SINGLE_INJECTOR_FOULING
        assert res5.sub_fault == "BANK1_CYL1"

        # FM_06: Valvetronic wear (vanishes at 9.7 mm)
        res6 = engine.isolate_faults(
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
        assert res6.primary_fault == FaultModeId.FM_06_VALVETRONIC_SHAFT_LEVER_WEAR
        assert res6.sub_fault == "CYL_1"

        # FM_07: Sticky VANOS solenoid
        res7 = engine.isolate_faults(
            MultiRegimeObservations(vanos_tracking_error_deg=1.45, vanos_slew_rate_deg_s=60.0, t_oil_c=88.0)
        )
        assert res7.primary_fault == FaultModeId.FM_07_STICKY_VANOS_SOLENOID

        # FM_08: Thermostat stuck open vs radiator fouling
        res8a = engine.isolate_faults(
            MultiRegimeObservations(t_coolant_c=72.0, t_rad_out_c=65.0, pwm_stat=0.0)
        )
        assert res8a.primary_fault == FaultModeId.FM_08_THERMOSTAT_OR_RADIATOR_FAULT
        assert res8a.sub_fault == ThermalSubFault.THERMOSTAT_STUCK_OPEN

        res8b = engine.isolate_faults(
            MultiRegimeObservations(t_coolant_c=112.0, t_rad_out_c=60.0, pwm_stat=1.0, theta_rad_health=0.45)
        )
        assert res8b.primary_fault == FaultModeId.FM_08_THERMOSTAT_OR_RADIATOR_FAULT
        assert res8b.sub_fault == ThermalSubFault.RADIATOR_FOULING_OR_CLOGGING
