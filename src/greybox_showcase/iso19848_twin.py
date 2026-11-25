"""ISO 19848:2018 Maritime Digital Twin Transferability & Polymorphic Abstraction.

Core Architectural Principles:
1. Universal Reciprocating Engine Data Contract:
   Adheres to ISO 19848:2018 (Standard data structures for shipboard machinery) and
   SFI Group 601 (Reciprocating Diesel/Otto Propulsion Machinery).
2. Pure Polymorphic Formulation (Zero Code Branching on Engine Name):
   Both high-revving 4-stroke automotive twins (BMW N42B20A) and ultra-long-stroke
   2-stroke low-speed marine propulsion twins (MAN B&W 6S50ME-C) execute the exact
   same underlying mathematical kernels across all 4 shared sub-models.
3. Dimensionless Similarity Engine:
   Computes Strouhal, Mach, Biot/NTU, and Fouling dimensionless similarity metrics
   proving mathematical transferability between automotive and maritime domains.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import yaml


# =============================================================================
# 1. ISO 19848:2018 & SFI Group 601 Channel Metadata & Telemetry Frame
# =============================================================================

class ISO19848ChannelMapping(BaseModel):
    """Channel definition conforming to ISO 19848:2018 Local Short IDs & SFI 601."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    iso_local_short_id: str = Field(
        ...,
        description="ISO 19848 URI path (e.g. /Propulsion/MainEngine/EngineSpeed)",
    )
    sfi_code: str = Field(
        ...,
        description="SFI Group 601 code (e.g. '601.11', '601.21', '601.31', '601.41')",
    )
    unit: str = Field(..., description="SI unit symbol (e.g. 'rpm', 'Pa', 'K', 'kg/s')")
    description: str = Field(default="", description="Human-readable channel description")
    native_signal: str = Field(default="", description="Native ECU / diagnostic register name")


class ISO19848TelemetryFrame(BaseModel):
    """Standardized SI Telemetry Frame conforming to ISO 19848:2018."""

    model_config = ConfigDict(extra="ignore")

    timestamp_iso: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp",
    )
    engine_speed_rpm: float = Field(..., ge=0.0, description="Rotational speed in RPM")
    manifold_or_scavenge_pressure_pa: float = Field(
        ..., gt=0.0, description="Charge-air receiver absolute pressure [Pa]"
    )
    charge_air_temperature_k: float = Field(
        ..., gt=0.0, description="Intake/scavenge air temperature [K]"
    )
    mass_air_flow_kg_s: float = Field(
        ..., ge=0.0, description="Mass air flow delivery rate [kg/s]"
    )
    valve_lift_m: float = Field(
        ..., ge=0.0, description="Variable valve actuation lift (Valvetronic / VVA) [m]"
    )
    coolant_temperature_k: float = Field(
        ..., gt=0.0, description="Engine block / Jacket Cooling Water temperature [K]"
    )
    oil_temperature_k: float = Field(
        ..., gt=0.0, description="Engine lubricating oil temperature [K]"
    )
    ambient_or_sink_temperature_k: float = Field(
        default=298.15, gt=0.0, description="Ambient / Seawater heat sink temperature [K]"
    )
    fuel_rail_pressure_pa: Optional[float] = Field(
        default=None, description="Fuel injection rail pressure [Pa]"
    )
    fuel_mass_flow_kg_s: Optional[float] = Field(
        default=None, description="Fuel mass flow consumption rate [kg/s]"
    )
    crankshaft_roughness_rad_s2: Optional[List[float]] = Field(
        default=None, description="Cylinder-specific rough running / IAS acceleration [rad/s^2]"
    )
    coolant_flow_fraction: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Coolant/seawater flow fraction [0, 1]"
    )
    oil_flow_fraction: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Oil pump flow fraction [0, 1]"
    )
    coolant_fouling_factor: Optional[float] = Field(
        default=None, ge=0.0, le=1.5, description="Direct online coolant fouling health factor theta_c in [0, 1.2]"
    )
    oil_fouling_factor: Optional[float] = Field(
        default=None, ge=0.0, le=1.5, description="Direct online oil fouling health factor theta_o in [0, 1.2]"
    )
    heat_input_coolant_w: Optional[float] = Field(
        default=None, description="Combustion heat flux rejected to block/coolant [W]"
    )
    heat_input_oil_w: Optional[float] = Field(
        default=None, description="Frictional/combustion heat flux rejected to oil [W]"
    )
    downstream_cylinder_pressure_pa: Optional[float] = Field(
        default=None, description="Downstream manifold or cylinder pressure for pressure ratio"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_config_path(path_or_str: Union[str, Path]) -> Path:
    """Resolves a config YAML path from CWD, repo config/, or bundled package data."""
    p = Path(path_or_str)
    if p.is_file():
        return p

    if not p.is_absolute():
        is_config_relative = len(p.parts) == 1 or (
            len(p.parts) == 2 and p.parts[0] == "config"
        )
        if is_config_relative:
            p_config = Path("config") / p.name
            if p_config.is_file():
                return p_config

            p_root_config = _REPO_ROOT / "config" / p.name
            if p_root_config.is_file():
                return p_root_config

        p_root_direct = _REPO_ROOT / p
        if p_root_direct.is_file():
            return p_root_direct

        if is_config_relative:
            pkg_configs = resources.files("greybox_showcase._configs")
            bundled = pkg_configs.joinpath(p.name)
            if bundled.is_file():
                if isinstance(bundled, Path):
                    return bundled
                candidate = Path(str(bundled))
                if candidate.is_file():
                    return candidate

    raise FileNotFoundError(f"Could not locate plant config YAML at '{path_or_str}'.")


# =============================================================================
# 2. Reciprocating Plant Configuration Model
# =============================================================================

class ReciprocatingPlantConfig(BaseModel):
    """Unified configuration for any reciprocating internal combustion engine."""

    model_config = ConfigDict(extra="ignore")

    schema_version: str = "3.1"
    standard: str = "ISO_19848_2018"
    sfi_group_code: str = "601"
    plant_metadata: Dict[str, Any] = Field(default_factory=dict)
    geometry_and_kinematics: Dict[str, Any] = Field(default_factory=dict)
    thermodynamics_and_breathing: Dict[str, Any] = Field(default_factory=dict)
    cooling_and_thermal: Dict[str, Any] = Field(default_factory=dict)
    injection_and_fuel_system: Dict[str, Any] = Field(default_factory=dict)
    qualification_and_diagnostics: Dict[str, Any] = Field(default_factory=dict)
    iso_19848_channel_mapping: Dict[str, ISO19848ChannelMapping] = Field(
        default_factory=dict
    )

    @classmethod
    def resolve_path(cls, path_or_str: Union[str, Path]) -> Path:
        """Resolves file path directly, against config/, repo root, or bundled package data."""
        return resolve_config_path(path_or_str)

    @classmethod
    def from_yaml(cls, path_or_str: Union[str, Path]) -> ReciprocatingPlantConfig:
        """Loads configuration from an ISO 19848 plant YAML file."""
        resolved = cls.resolve_path(path_or_str)
        with open(resolved, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"Invalid YAML content in {resolved}")

        return cls(**raw)


# =============================================================================
# 3. Subsystem Evaluation Snapshots
# =============================================================================

@dataclass
class SubModel1BreathingSnapshot:
    """Sub-Model 1: Compressible Orifice Flow & Trapped Charge Ingestion."""

    pressure_ratio_pi: float
    smooth_pi: float
    flow_function_psi: float
    m_dot_orifice_kg_s: float
    m_dot_trapped_kg_s: float
    m_dot_calc_kg_s: float
    air_mass_residual_kg_s: float
    choked: bool
    sfi_code: str = "601.11"


@dataclass
class SubModel2ThermalSnapshot:
    """Sub-Model 2: Coupled 2-Node Thermal ODE & Heat-Exchanger Fouling Factor."""

    q_cool_flux_w: float
    q_oil_flux_w: float
    d_temp_coolant_dt_k_s: float
    d_temp_oil_dt_k_s: float
    actual_ua_coolant_w_per_k: float
    actual_ua_oil_w_per_k: float
    fouling_factor_coolant: float
    fouling_factor_oil: float
    fouling_alert: bool
    sfi_code: str = "601.41"


@dataclass
class SubModel3FuelTrimSnapshot:
    """Sub-Model 3: Stoichiometric/SFOC & Injection Anomaly Decomposition."""

    unmetered_leak_kg_s: float
    fuel_pressure_droop_ratio: float
    individual_cylinder_injection_deviations: List[float]
    sfoc_or_fuel_trim_residual: float
    leak_detected: bool
    sfi_code: str = "601.21"


@dataclass
class SubModel4IASRoughnessSnapshot:
    """Sub-Model 4: Crankshaft Instantaneous Angular Speed (IAS) Harmonic Analysis."""

    firing_order: List[int]
    raw_roughness: List[float]
    firing_sequence_roughness: List[float]
    peak_cylinder: int
    max_spread_rad_s2: float
    order_harmonics_mag: Dict[str, float]
    order_harmonics_phase_rad: Dict[str, float]
    gate4_passed: bool
    imbalance_severity: float
    balancing_corrections: List[float]
    sfi_code: str = "601.31"


@dataclass
class TwinEvaluationSnapshot:
    """Unified step evaluation snapshot combining all 4 isomorphic subsystems."""

    timestamp_iso: str
    plant_id: str
    submodel1_breathing: SubModel1BreathingSnapshot
    submodel2_thermal: SubModel2ThermalSnapshot
    submodel3_fuel_trim: SubModel3FuelTrimSnapshot
    submodel4_ias_roughness: SubModel4IASRoughnessSnapshot
    dimensionless_numbers: Dict[str, float]
    summary: str


# =============================================================================
# 4. Universal Abstract Reciprocating Twin Class
# =============================================================================

class AbstractReciprocatingTwin:
    """Pure Polymorphic Grey-Box Digital Twin for Reciprocating Engines (ISO 19848 / SFI 601).

    Instantiable directly via:
      twin = AbstractReciprocatingTwin("bmw_n42b20a.yaml")
      twin = AbstractReciprocatingTwin("maritime_man_bw_6s50me_c.yaml")
    with strictly zero code changes or engine name branching.
    """

    def __init__(
        self,
        config_source: Union[str, Path, Dict[str, Any], ReciprocatingPlantConfig],
    ) -> None:
        if isinstance(config_source, ReciprocatingPlantConfig):
            self.config = config_source
        elif isinstance(config_source, dict):
            self.config = ReciprocatingPlantConfig(**config_source)
        elif isinstance(config_source, (str, Path)):
            self.config = ReciprocatingPlantConfig.from_yaml(config_source)
        else:
            raise TypeError(f"Unsupported config source type: {type(config_source)}")

        self.geo = self.config.geometry_and_kinematics
        self.thermo = self.config.thermodynamics_and_breathing
        self.thermal = self.config.cooling_and_thermal
        self.fuel_sys = self.config.injection_and_fuel_system
        self.qual = self.config.qualification_and_diagnostics
        self.meta = self.config.plant_metadata

        self.plant_id: str = self.meta.get("plant_id", "RECIPROCATING_PLANT")
        self.cylinders: int = int(self.geo.get("cylinders", 4))
        self.firing_order: List[int] = list(
            self.geo.get("firing_order", list(range(1, self.cylinders + 1)))
        )
        self.strokes_per_cycle: int = int(
            self.geo.get(
                "strokes_per_cycle",
                self.meta.get(
                    "strokes_per_cycle",
                    2 if "2_stroke" in str(self.meta.get("engine_cycle", "")) else 4,
                ),
            )
        )
        self.displacement_m3: float = float(self.geo.get("displacement_m3", 0.002))
        self.bore_m: float = float(self.geo.get("bore_m", 0.084))
        self.stroke_m: float = float(self.geo.get("stroke_m", 0.090))

    @classmethod
    def from_yaml(cls, path_or_str: Union[str, Path]) -> AbstractReciprocatingTwin:
        """Factory constructor loading configuration from YAML path or filename."""
        return cls(config_source=path_or_str)

    # -------------------------------------------------------------------------
    # Sub-Model 1: Compressible Saint-Venant/Wantzel Flow & Cylinder Trapping
    # -------------------------------------------------------------------------
    def evaluate_submodel1_breathing(
        self,
        frame: ISO19848TelemetryFrame,
    ) -> SubModel1BreathingSnapshot:
        """Computes orifice flow, trapped mass, and smooth p-norm blend."""
        p_up = frame.manifold_or_scavenge_pressure_pa
        t_up = max(frame.charge_air_temperature_k, 200.0)
        l_valve = max(frame.valve_lift_m, 1e-6)
        rpm = max(frame.engine_speed_rpm, 0.0)

        if frame.downstream_cylinder_pressure_pa is not None:
            p_down = frame.downstream_cylinder_pressure_pa
        else:
            l_max = float(self.geo.get("valve_lift_max_m", 0.010))
            lift_ratio = min(max(l_valve / l_max, 0.02), 1.0)
            p_down = p_up * (0.35 + 0.60 * lift_ratio)

        gamma = float(self.thermo.get("gamma", 1.4))
        r_spec = float(self.thermo.get("r_specific", 287.05))
        pi_crit = float(self.thermo.get("critical_pressure_ratio_pi", 0.5283))
        eps_pi = float(self.thermo.get("softplus_eps_pi", 0.001))
        kp = float(self.thermo.get("p_norm_blend_kp", 6.0))
        cd = float(self.thermo.get("discharge_coeff_cd", 0.70))

        # 1. Smooth Softplus Pressure Ratio (C^infinity across sonic transition)
        pi_raw = min(max(p_down / max(p_up, 1.0), 1e-4), 0.9999)
        smooth_pi = pi_crit + 0.5 * (
            (pi_raw - pi_crit) + math.sqrt((pi_raw - pi_crit) ** 2 + eps_pi ** 2)
        )
        smooth_pi = min(smooth_pi, 0.9999)

        # 2. Saint-Venant & Wantzel flow function Psi_tilde(Pi)
        exponent_1 = 2.0 / gamma
        exponent_2 = (gamma + 1.0) / gamma
        radicand = (2.0 * gamma / (gamma - 1.0)) * (
            smooth_pi ** exponent_1 - smooth_pi ** exponent_2
        )
        flow_psi = math.sqrt(max(radicand, 1e-12))
        choked = bool(pi_raw <= pi_crit)

        # 3. Curtain Orifice Area and Flow
        d_valve = float(self.geo.get("intake_valve_diameter_m", 0.032))
        n_valves = int(self.geo.get("intake_valves_per_cyl", 2))
        curtain_area_per_cyl = n_valves * math.pi * d_valve * l_valve
        total_curtain_area = curtain_area_per_cyl * self.cylinders

        m_dot_orifice = cd * total_curtain_area * (p_up / math.sqrt(r_spec * t_up)) * flow_psi

        # 4. Trapped Cylinder Ingestion
        strokes = self.strokes_per_cycle
        firings_per_sec = (rpm / 60.0) / (strokes / 2.0) * self.cylinders
        v_eff = (self.displacement_m3 / self.cylinders) * float(
            self.thermo.get("volumetric_efficiency_base", 0.92)
        )
        rho_charge = p_up / (r_spec * t_up)
        m_dot_trapped = firings_per_sec * v_eff * rho_charge

        m_dot_orifice = max(m_dot_orifice, 1e-8)
        m_dot_trapped = max(m_dot_trapped, 1e-8)

        # 5. Smooth p-Norm Blend: ((m_dot_orif)^-kp + (m_dot_trap)^-kp)^(-1/kp)
        m_dot_calc = ((m_dot_orifice ** (-kp)) + (m_dot_trapped ** (-kp))) ** (-1.0 / kp)
        residual = frame.mass_air_flow_kg_s - m_dot_calc

        return SubModel1BreathingSnapshot(
            pressure_ratio_pi=pi_raw,
            smooth_pi=smooth_pi,
            flow_function_psi=flow_psi,
            m_dot_orifice_kg_s=m_dot_orifice,
            m_dot_trapped_kg_s=m_dot_trapped,
            m_dot_calc_kg_s=m_dot_calc,
            air_mass_residual_kg_s=residual,
            choked=choked,
            sfi_code="601.11",
        )

    # -------------------------------------------------------------------------
    # Sub-Model 2: Coupled 2-Node Thermal ODE & Heat-Exchanger Fouling Factor
    # -------------------------------------------------------------------------
    def evaluate_submodel2_thermal(
        self,
        frame: ISO19848TelemetryFrame,
    ) -> SubModel2ThermalSnapshot:
        """Computes coolant/oil enthalpy balance and online heat-exchanger fouling factor."""
        t_c = frame.coolant_temperature_k
        t_o = frame.oil_temperature_k
        t_sink = frame.ambient_or_sink_temperature_k

        c_th_c = float(self.thermal.get("capacitance_coolant_j_per_k", 92000.0))
        c_th_o = float(self.thermal.get("capacitance_oil_j_per_k", 18500.0))
        ua_c_clean = float(self.thermal.get("ua_coolant_clean_w_per_k", 1650.0))
        ua_o_clean = float(self.thermal.get("ua_oil_clean_w_per_k", 380.0))
        ua_amb_c = float(self.thermal.get("ua_ambient_coolant_w_per_k", 45.0))
        ua_amb_o = float(self.thermal.get("ua_ambient_oil_w_per_k", 25.0))
        fouling_thresh = float(self.thermal.get("fouling_alarm_threshold", 0.65))

        u_c = min(max(frame.coolant_flow_fraction, 0.05), 1.0)
        u_o = min(max(frame.oil_flow_fraction, 0.05), 1.0)

        if frame.heat_input_coolant_w is not None:
            q_in_c = frame.heat_input_coolant_w
        else:
            if frame.fuel_mass_flow_kg_s is not None:
                fuel_flow = frame.fuel_mass_flow_kg_s
            else:
                fuel_flow = frame.mass_air_flow_kg_s / float(
                    self.thermo.get("air_fuel_ratio_stoich", 14.7)
                )
            lhv = float(self.thermo.get("fuel_lhv_j_per_kg", 43400000.0))
            q_in_c = 0.28 * fuel_flow * lhv

        if frame.heat_input_oil_w is not None:
            q_in_o = frame.heat_input_oil_w
        else:
            q_in_o = 0.08 * q_in_c

        delta_t_c = max(t_c - t_sink, 0.0)
        delta_t_o = max(t_o - t_c, -50.0)

        if frame.coolant_fouling_factor is not None:
            fouling_c = float(np.clip(frame.coolant_fouling_factor, 0.0, 1.2))
            ua_c_actual = ua_c_clean * fouling_c * u_c
        else:
            ua_c_actual = ua_c_clean * u_c
            fouling_c = float(np.clip(ua_c_actual / max(ua_c_clean, 1e-6), 0.0, 1.2))

        if frame.oil_fouling_factor is not None:
            fouling_o = float(np.clip(frame.oil_fouling_factor, 0.0, 1.2))
            ua_o_actual = ua_o_clean * fouling_o * u_o
        else:
            ua_o_actual = ua_o_clean * u_o
            fouling_o = float(np.clip(ua_o_actual / max(ua_o_clean, 1e-6), 0.0, 1.2))

        q_cool = ua_c_actual * delta_t_c
        q_oil = ua_o_actual * delta_t_o
        q_amb_c = ua_amb_c * (t_c - t_sink)
        q_amb_o = ua_amb_o * (t_o - t_sink)

        dt_c_dt = (q_in_c - q_cool - q_oil - q_amb_c) / c_th_c
        dt_o_dt = (q_in_o + q_oil - q_amb_o) / c_th_o

        fouling_alert = bool(fouling_c < fouling_thresh or fouling_o < fouling_thresh)

        return SubModel2ThermalSnapshot(
            q_cool_flux_w=q_cool,
            q_oil_flux_w=q_oil,
            d_temp_coolant_dt_k_s=dt_c_dt,
            d_temp_oil_dt_k_s=dt_o_dt,
            actual_ua_coolant_w_per_k=ua_c_actual,
            actual_ua_oil_w_per_k=ua_o_actual,
            fouling_factor_coolant=fouling_c,
            fouling_factor_oil=fouling_o,
            fouling_alert=fouling_alert,
            sfi_code="601.41",
        )

    # -------------------------------------------------------------------------
    # Sub-Model 3: Stoichiometric/SFOC & Injection Anomaly Decomposition
    # -------------------------------------------------------------------------
    def evaluate_submodel3_fuel_trim(
        self,
        frame: ISO19848TelemetryFrame,
    ) -> SubModel3FuelTrimSnapshot:
        """Decomposes manifold/scavenge air leaks from fuel rail droop and injector bias."""
        p_up = frame.manifold_or_scavenge_pressure_pa
        p_amb = float(self.thermo.get("ambient_pressure_pa", 101325.0))

        delta_p_leak = abs(p_amb - p_up)
        leak_conductance_est = 2.5e-6  # m^2 equivalent orifice
        m_dot_leak = leak_conductance_est * math.sqrt(max(delta_p_leak, 0.0))

        p_rail_nominal = float(self.fuel_sys.get("nominal_fuel_rail_pressure_pa", 350000.0))
        p_rail_actual = (
            frame.fuel_rail_pressure_pa
            if frame.fuel_rail_pressure_pa is not None
            else p_rail_nominal
        )
        rail_droop_ratio = float((p_rail_actual - p_rail_nominal) / max(p_rail_nominal, 1.0))

        cyl_deviations: List[float] = []
        if frame.crankshaft_roughness_rad_s2 and len(frame.crankshaft_roughness_rad_s2) == self.cylinders:
            mean_rr = float(np.mean(frame.crankshaft_roughness_rad_s2))
            for rr in frame.crankshaft_roughness_rad_s2:
                cyl_deviations.append(float(-0.05 * (rr - mean_rr)))
        else:
            cyl_deviations = [0.0] * self.cylinders

        m_dot_air = max(frame.mass_air_flow_kg_s, 1e-6)
        sfoc_residual = float(
            (m_dot_leak / m_dot_air) - 0.5 * rail_droop_ratio + float(np.mean(cyl_deviations))
        )
        leak_detected = bool(m_dot_leak > 0.0005)

        return SubModel3FuelTrimSnapshot(
            unmetered_leak_kg_s=m_dot_leak,
            fuel_pressure_droop_ratio=rail_droop_ratio,
            individual_cylinder_injection_deviations=cyl_deviations,
            sfoc_or_fuel_trim_residual=sfoc_residual,
            leak_detected=leak_detected,
            sfi_code="601.21",
        )

    # -------------------------------------------------------------------------
    # Sub-Model 4: Crankshaft Instantaneous Angular Speed (IAS) Harmonic Analysis
    # -------------------------------------------------------------------------
    def evaluate_submodel4_ias(
        self,
        frame: ISO19848TelemetryFrame,
    ) -> SubModel4IASRoughnessSnapshot:
        """Arbitrary N-cylinder firing-order Fourier harmonic order analysis."""
        n_cyl = self.cylinders
        firing_order = self.firing_order

        if frame.crankshaft_roughness_rad_s2 and len(frame.crankshaft_roughness_rad_s2) == n_cyl:
            raw_rr = [float(x) for x in frame.crankshaft_roughness_rad_s2]
        else:
            raw_rr = [0.0] * n_cyl

        y_firing = [raw_rr[c - 1] for c in firing_order]

        abs_rr = [abs(x) for x in raw_rr]
        max_idx = int(np.argmax(abs_rr))
        peak_cyl = max_idx + 1
        max_spread = float(abs_rr[max_idx])

        gate4_max = float(self.qual.get("gate4_max_rough_running_rad_s2", 0.8))
        gate4_passed = bool(max_spread < gate4_max)

        angles = np.array([m * (2.0 * math.pi / n_cyl) for m in range(n_cyl)], dtype=np.float64)
        y_arr = np.array(y_firing, dtype=np.float64)

        harmonics_mag: Dict[str, float] = {}
        harmonics_phase: Dict[str, float] = {}

        a_1 = float((2.0 / n_cyl) * np.sum(y_arr * np.cos(angles)))
        b_1 = float((2.0 / n_cyl) * np.sum(y_arr * np.sin(angles)))
        mag_1 = float(math.sqrt(a_1 * a_1 + b_1 * b_1))
        phase_1 = float(math.atan2(b_1, a_1))

        a_2 = float((2.0 / n_cyl) * np.sum(y_arr * np.cos(2.0 * angles)))
        b_2 = float((2.0 / n_cyl) * np.sum(y_arr * np.sin(2.0 * angles)))
        mag_2 = float(math.sqrt(a_2 * a_2 + b_2 * b_2))
        phase_2 = float(math.atan2(b_2, a_2))

        mean_y = float(np.mean(y_arr))
        firing_pulsation = float(math.sqrt(float(np.mean((y_arr - mean_y) ** 2))))

        if self.strokes_per_cycle == 4:
            harmonics_mag["order_0_5"] = mag_1
            harmonics_phase["order_0_5"] = phase_1
            harmonics_mag["order_1_0"] = mag_2
            harmonics_phase["order_1_0"] = phase_2
            harmonics_mag["order_firing"] = firing_pulsation
        else:
            harmonics_mag["order_1_0"] = mag_1
            harmonics_phase["order_1_0"] = phase_1
            harmonics_mag["order_2_0"] = mag_2
            harmonics_phase["order_2_0"] = phase_2
            harmonics_mag["order_firing"] = firing_pulsation

        balancing_corrections = [float(-0.5 * (val - mean_y)) for val in raw_rr]
        imbalance_severity = float(np.std(raw_rr))

        return SubModel4IASRoughnessSnapshot(
            firing_order=firing_order,
            raw_roughness=raw_rr,
            firing_sequence_roughness=y_firing,
            peak_cylinder=peak_cyl,
            max_spread_rad_s2=max_spread,
            order_harmonics_mag=harmonics_mag,
            order_harmonics_phase_rad=harmonics_phase,
            gate4_passed=gate4_passed,
            imbalance_severity=imbalance_severity,
            balancing_corrections=balancing_corrections,
            sfi_code="601.31",
        )

    # -------------------------------------------------------------------------
    # Full Unified Step Evaluation
    # -------------------------------------------------------------------------
    def evaluate_step(
        self,
        frame: ISO19848TelemetryFrame,
    ) -> TwinEvaluationSnapshot:
        """Executes full multi-subsystem grey-box evaluation step."""
        sm1 = self.evaluate_submodel1_breathing(frame)
        sm2 = self.evaluate_submodel2_thermal(frame)
        sm3 = self.evaluate_submodel3_fuel_trim(frame)
        sm4 = self.evaluate_submodel4_ias(frame)

        pi_val = sm1.pressure_ratio_pi
        gamma = float(self.thermo.get("gamma", 1.4))
        if pi_val < 0.999:
            mach = float(
                math.sqrt((2.0 / (gamma - 1.0)) * (pi_val ** (-(gamma - 1.0) / gamma) - 1.0))
            )
        else:
            mach = 0.01

        rpm = max(frame.engine_speed_rpm, 1.0)
        v_piston = 2.0 * self.stroke_m * (rpm / 60.0)
        firings_hz = (rpm / 60.0) / (self.strokes_per_cycle / 2.0) * self.cylinders
        strouhal = float((firings_hz * self.bore_m) / max(v_piston, 0.1))

        tau_thermal = 22.0
        c_th_c = float(self.thermal.get("capacitance_coolant_j_per_k", 92000.0))
        ntu_coolant = float((sm2.actual_ua_coolant_w_per_k * tau_thermal) / c_th_c)

        gate4_max = float(self.qual.get("gate4_max_rough_running_rad_s2", 0.8))
        roughness_z = float(sm4.max_spread_rad_s2 / max(gate4_max, 0.01))

        dimensionless = {
            "pressure_ratio_pi": pi_val,
            "orifice_mach_number": mach,
            "strouhal_number": strouhal,
            "ntu_coolant": ntu_coolant,
            "fouling_factor_coolant": sm2.fouling_factor_coolant,
            "roughness_z_score": roughness_z,
        }

        summary = (
            f"Twin [{self.plant_id}] Step @ {rpm:.1f} RPM: m_dot={sm1.m_dot_calc_kg_s:.4f} kg/s, "
            f"T_c={frame.coolant_temperature_k:.1f} K, Gate 4 {'PASS' if sm4.gate4_passed else 'ALERT'} "
            f"(Peak Cyl {sm4.peak_cylinder}, spread={sm4.max_spread_rad_s2:.3f} rad/s^2), "
            f"Fouling theta_c={sm2.fouling_factor_coolant:.2f}."
        )

        return TwinEvaluationSnapshot(
            timestamp_iso=frame.timestamp_iso,
            plant_id=self.plant_id,
            submodel1_breathing=sm1,
            submodel2_thermal=sm2,
            submodel3_fuel_trim=sm3,
            submodel4_ias_roughness=sm4,
            dimensionless_numbers=dimensionless,
            summary=summary,
        )

    # -------------------------------------------------------------------------
    # Dimensionless Similarity Engine
    # -------------------------------------------------------------------------
    def compute_dimensionless_similarity(
        self,
        other_twin: AbstractReciprocatingTwin,
    ) -> Dict[str, float]:
        """Calculates dimensionless similarity numbers between two reciprocating twins."""
        rpm_self = float(self.meta.get("rated_speed_rpm", 6000.0)) * 0.4
        rpm_other = float(other_twin.meta.get("rated_speed_rpm", 127.0)) * 0.7

        v_piston_self = 2.0 * self.stroke_m * (rpm_self / 60.0)
        v_piston_other = 2.0 * other_twin.stroke_m * (rpm_other / 60.0)

        f_self = (rpm_self / 60.0) / (self.strokes_per_cycle / 2.0) * self.cylinders
        f_other = (rpm_other / 60.0) / (other_twin.strokes_per_cycle / 2.0) * other_twin.cylinders

        strouhal_self = (f_self * self.bore_m) / max(v_piston_self, 0.1)
        strouhal_other = (f_other * other_twin.bore_m) / max(v_piston_other, 0.1)

        strouhal_ratio = float(strouhal_self / max(strouhal_other, 1e-6))

        pi_crit_self = float(self.thermo.get("critical_pressure_ratio_pi", 0.5283))
        pi_crit_other = float(other_twin.thermo.get("critical_pressure_ratio_pi", 0.5283))
        mach_parity = float(abs(pi_crit_self - pi_crit_other))

        c_c_self = float(self.thermal.get("capacitance_coolant_j_per_k", 92000.0))
        c_c_other = float(other_twin.thermal.get("capacitance_coolant_j_per_k", 4800000.0))
        ua_c_self = float(self.thermal.get("ua_coolant_clean_w_per_k", 1650.0))
        ua_c_other = float(other_twin.thermal.get("ua_coolant_clean_w_per_k", 120000.0))

        rate_self = ua_c_self / c_c_self
        rate_other = ua_c_other / c_c_other
        cooling_dynamics_scale_ratio = float(rate_self / max(rate_other, 1e-6))

        firing_symmetry_parity = (
            1.0 if (self.cylinders % 2 == 0 and other_twin.cylinders % 2 == 0) else 0.5
        )

        return {
            "strouhal_number_self": float(strouhal_self),
            "strouhal_number_other": float(strouhal_other),
            "strouhal_scale_ratio": strouhal_ratio,
            "mach_critical_discrepancy": mach_parity,
            "cooling_dynamics_scale_ratio": cooling_dynamics_scale_ratio,
            "firing_symmetry_parity": firing_symmetry_parity,
            "isomorphic_transferability_score": float(
                np.clip(1.0 - abs(math.log10(max(strouhal_ratio, 1e-4))), 0.0, 1.0)
            ),
        }
