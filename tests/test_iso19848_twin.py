"""Unit tests for ISO 19848:2018 polymorphic twin (Automotive <-> Maritime)."""

from __future__ import annotations

from importlib import resources
import math
from pathlib import Path

import pytest

import greybox_showcase
import greybox_showcase.iso19848_twin as iso19848_twin
from greybox_showcase.iso19848_twin import (
    AbstractReciprocatingTwin,
    ISO19848TelemetryFrame,
    TwinEvaluationSnapshot,
    resolve_config_path,
)
from greybox_showcase.observability import load_fault_isolation_matrix


class TestISO19848DualInstantiation:
    """Tests zero-code-change polymorphic instantiation of AbstractReciprocatingTwin."""

    def test_instantiate_bmw_n42b20a(self) -> None:
        twin = AbstractReciprocatingTwin("bmw_n42b20a.yaml")
        assert twin.plant_id == "BMW_N42B20A"
        assert twin.cylinders == 4
        assert twin.firing_order == [1, 3, 4, 2]
        assert twin.strokes_per_cycle == 4
        assert math.isclose(twin.displacement_m3, 0.001995, rel_tol=1e-3)
        assert math.isclose(twin.bore_m, 0.084, abs_tol=1e-4)
        assert math.isclose(twin.stroke_m, 0.090, abs_tol=1e-4)

    def test_instantiate_maritime_man_bw_6s50me_c(self) -> None:
        twin = AbstractReciprocatingTwin("maritime_man_bw_6s50me_c.yaml")
        assert twin.plant_id == "MAN_BW_6S50ME_C"
        assert twin.cylinders == 6
        assert twin.firing_order == [1, 5, 3, 4, 2, 6]
        assert twin.strokes_per_cycle == 2
        assert math.isclose(twin.displacement_m3, 2.3562, rel_tol=1e-3)
        assert math.isclose(twin.bore_m, 0.500, abs_tol=1e-4)
        assert math.isclose(twin.stroke_m, 2.000, abs_tol=1e-4)

    def test_path_resolution_variants(self) -> None:
        t1 = AbstractReciprocatingTwin.from_yaml("config/bmw_n42b20a.yaml")
        t2 = AbstractReciprocatingTwin.from_yaml(Path("config/maritime_man_bw_6s50me_c.yaml"))
        assert t1.cylinders == 4
        assert t2.cylinders == 6

    def test_bundled_configs_present_and_synced(self) -> None:
        repo_cfg_dir = Path(__file__).resolve().parent.parent / "config"
        pkg_configs = resources.files("greybox_showcase._configs")
        for filename in (
            "bmw_n42b20a.yaml",
            "maritime_man_bw_6s50me_c.yaml",
            "fault_isolation_matrix.yaml",
        ):
            bundled = pkg_configs.joinpath(filename)
            assert bundled.is_file()
            if repo_cfg_dir.is_dir():
                assert bundled.read_text(encoding="utf-8") == (
                    repo_cfg_dir / filename
                ).read_text(encoding="utf-8")

    def test_bundled_config_resolution_outside_repo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(iso19848_twin, "_REPO_ROOT", tmp_path)

        resolved_bmw = resolve_config_path("bmw_n42b20a.yaml")
        assert resolved_bmw.is_file()
        assert resolved_bmw.parent.name == "_configs"

        bmw = AbstractReciprocatingTwin("bmw_n42b20a.yaml")
        marine = AbstractReciprocatingTwin("config/maritime_man_bw_6s50me_c.yaml")
        matrix = load_fault_isolation_matrix("fault_isolation_matrix.yaml")

        assert bmw.plant_id == "BMW_N42B20A"
        assert marine.plant_id == "MAN_BW_6S50ME_C"
        assert len(matrix["fault_modes"]) == 8

    def test_package_version_updated(self) -> None:
        assert greybox_showcase.__version__ == "0.1.1"


class TestSubModelIsomorphism:
    """Tests all 4 shared physical sub-models across automotive and marine twins."""

    def test_submodel1_breathing_choking_and_p_norm(self) -> None:
        twin = AbstractReciprocatingTwin("bmw_n42b20a.yaml")
        frame_choked = ISO19848TelemetryFrame(
            engine_speed_rpm=700.0,
            manifold_or_scavenge_pressure_pa=100000.0,
            charge_air_temperature_k=300.0,
            mass_air_flow_kg_s=0.0035,
            valve_lift_m=0.00035,
            coolant_temperature_k=378.15,
            oil_temperature_k=368.15,
            downstream_cylinder_pressure_pa=45000.0,
        )
        snap = twin.evaluate_submodel1_breathing(frame_choked)
        assert snap.choked is True
        assert snap.pressure_ratio_pi < 0.5283
        assert snap.m_dot_calc_kg_s > 0.0
        assert snap.sfi_code == "601.11"

    def test_submodel1_maritime_scavenge_flow(self) -> None:
        twin = AbstractReciprocatingTwin("maritime_man_bw_6s50me_c.yaml")
        frame_mcr = ISO19848TelemetryFrame(
            engine_speed_rpm=127.0,
            manifold_or_scavenge_pressure_pa=320000.0,
            charge_air_temperature_k=315.15,
            mass_air_flow_kg_s=25.0,
            valve_lift_m=0.1200,
            coolant_temperature_k=358.15,
            oil_temperature_k=328.15,
        )
        snap = twin.evaluate_submodel1_breathing(frame_mcr)
        assert 0.0 < snap.m_dot_calc_kg_s < 50.0
        assert snap.sfi_code == "601.11"

    def test_submodel2_thermal_fouling_alert(self) -> None:
        twin_marine = AbstractReciprocatingTwin("maritime_man_bw_6s50me_c.yaml")
        f_fouled = ISO19848TelemetryFrame(
            engine_speed_rpm=127.0,
            manifold_or_scavenge_pressure_pa=320000.0,
            charge_air_temperature_k=315.15,
            mass_air_flow_kg_s=25.0,
            valve_lift_m=0.12,
            coolant_temperature_k=365.15,
            oil_temperature_k=335.15,
            coolant_flow_fraction=0.55,
        )
        snap = twin_marine.evaluate_submodel2_thermal(f_fouled)
        assert snap.fouling_alert is True
        assert snap.fouling_factor_coolant < 0.65
        assert snap.sfi_code == "601.41"

    def test_submodel3_fuel_rail_droop(self) -> None:
        twin = AbstractReciprocatingTwin("maritime_man_bw_6s50me_c.yaml")
        frame_droop = ISO19848TelemetryFrame(
            engine_speed_rpm=127.0,
            manifold_or_scavenge_pressure_pa=320000.0,
            charge_air_temperature_k=315.15,
            mass_air_flow_kg_s=25.0,
            valve_lift_m=0.12,
            coolant_temperature_k=358.15,
            oil_temperature_k=328.15,
            fuel_rail_pressure_pa=72000000.0,
        )
        snap = twin.evaluate_submodel3_fuel_trim(frame_droop)
        assert math.isclose(snap.fuel_pressure_droop_ratio, -0.10, abs_tol=1e-3)
        assert snap.sfi_code == "601.21"

    def test_submodel4_ias_4cyl_and_6cyl(self) -> None:
        twin_bmw = AbstractReciprocatingTwin("bmw_n42b20a.yaml")
        snap4 = twin_bmw.evaluate_submodel4_ias(
            ISO19848TelemetryFrame(
                engine_speed_rpm=700.0,
                manifold_or_scavenge_pressure_pa=96325.0,
                charge_air_temperature_k=305.15,
                mass_air_flow_kg_s=0.0035,
                valve_lift_m=0.00035,
                coolant_temperature_k=378.15,
                oil_temperature_k=368.15,
                crankshaft_roughness_rad_s2=[1.50, -0.05, 0.02, -0.01],
            )
        )
        assert snap4.peak_cylinder == 1
        assert snap4.gate4_passed is False
        assert snap4.order_harmonics_mag["order_0_5"] > 0.5

        twin_marine = AbstractReciprocatingTwin("maritime_man_bw_6s50me_c.yaml")
        snap6 = twin_marine.evaluate_submodel4_ias(
            ISO19848TelemetryFrame(
                engine_speed_rpm=127.0,
                manifold_or_scavenge_pressure_pa=320000.0,
                charge_air_temperature_k=315.15,
                mass_air_flow_kg_s=25.0,
                valve_lift_m=0.12,
                coolant_temperature_k=358.15,
                oil_temperature_k=328.15,
                crankshaft_roughness_rad_s2=[0.01, -0.02, 0.01, 0.02, 0.55, -0.01],
            )
        )
        assert snap6.peak_cylinder == 5
        assert snap6.gate4_passed is False
        assert len(snap6.balancing_corrections) == 6

    def test_dimensionless_similarity_and_full_step(self) -> None:
        twin_bmw = AbstractReciprocatingTwin("bmw_n42b20a.yaml")
        twin_marine = AbstractReciprocatingTwin("maritime_man_bw_6s50me_c.yaml")

        sim = twin_bmw.compute_dimensionless_similarity(twin_marine)
        assert sim["mach_critical_discrepancy"] == 0.0
        assert 0.1 < sim["strouhal_scale_ratio"] < 10.0
        assert sim["isomorphic_transferability_score"] > 0.50

        snap = twin_bmw.evaluate_step(
            ISO19848TelemetryFrame(
                engine_speed_rpm=700.0,
                manifold_or_scavenge_pressure_pa=96325.0,
                charge_air_temperature_k=305.15,
                mass_air_flow_kg_s=0.0035,
                valve_lift_m=0.00035,
                coolant_temperature_k=378.15,
                oil_temperature_k=368.15,
                crankshaft_roughness_rad_s2=[0.05, -0.02, 0.04, -0.01],
            )
        )
        assert isinstance(snap, TwinEvaluationSnapshot)
        assert "strouhal_number" in snap.dimensionless_numbers

    def test_zero_fuel_flow_dfco_and_zero_rail_pressure(self) -> None:
        twin = AbstractReciprocatingTwin("bmw_n42b20a.yaml")

        # Deceleration Fuel Cut-Off (DFCO): fuel_mass_flow_kg_s == 0.0 despite air flow > 0
        frame_dfco = ISO19848TelemetryFrame(
            engine_speed_rpm=2500.0,
            manifold_or_scavenge_pressure_pa=45000.0,
            charge_air_temperature_k=300.0,
            mass_air_flow_kg_s=0.015,
            valve_lift_m=0.005,
            coolant_temperature_k=368.15,
            oil_temperature_k=368.15,
            ambient_or_sink_temperature_k=298.15,
            fuel_mass_flow_kg_s=0.0,
            fuel_rail_pressure_pa=0.0,
        )
        thermal_dfco = twin.evaluate_submodel2_thermal(frame_dfco)
        # With q_in_c = 0.0 and T_coolant > T_ambient, engine must cool down (negative derivative)
        assert thermal_dfco.d_temp_coolant_dt_k_s < 0.0
        # Compare against fuel_mass_flow_kg_s=None where stoichiometric fallback adds positive heat
        frame_stoich = frame_dfco.model_copy(update={"fuel_mass_flow_kg_s": None})
        thermal_stoich = twin.evaluate_submodel2_thermal(frame_stoich)
        c_th_c = float(twin.thermal["capacitance_coolant_j_per_k"])
        lhv = float(twin.thermo["fuel_lhv_j_per_kg"])
        afr = float(twin.thermo["air_fuel_ratio_stoich"])
        expected_delta_q = 0.28 * (frame_dfco.mass_air_flow_kg_s / afr) * lhv
        assert math.isclose(
            (thermal_stoich.d_temp_coolant_dt_k_s - thermal_dfco.d_temp_coolant_dt_k_s) * c_th_c,
            expected_delta_q,
            rel_tol=1e-9,
        )

        # Depressurized rail: fuel_rail_pressure_pa == 0.0 -> 100% droop (-1.0), not nominal (0.0)
        fuel_zero_rail = twin.evaluate_submodel3_fuel_trim(frame_dfco)
        assert math.isclose(fuel_zero_rail.fuel_pressure_droop_ratio, -1.0, abs_tol=1e-12)
