"""8-Fault Multi-Regime Deterministic Isolation Matrix & Jacobian Rank Gate.

Governing Principles:
1. Multi-Regime Sensitivity Jacobian H & Normalized Observability Matrix H_tilde:
   In a single Valvetronic idle regime (Delta P = 50 mbar, L_vvt = 0.35 mm), leak flow,
   MAF gain, MAF offset, and fuel pressure drop are mathematically collinear (rank = 4 < 8,
   cond(H_tilde) = inf).
   Stacking 4 excitation regimes:
     (1) Stage 1 KOEO (RPM = 0, V_HFM in [0.97, 1.03] V)
     (2) Stage 2 Valvetronic Idle (50 mbar vacuum, micro-lift 0.35 mm)
     (3) Stage 3 Active Throttled Idle (500 mbar vacuum, high-lift 9.7 mm)
     (4) Stage 4 Cruise & Dynamic Slew / Warmup
   yields full column rank (rank = 8) and an equilibrated observability matrix with
   cond(H_tilde) = 6.413 < 50.0.

2. Crankshaft Angular Speed (Delta omega_k) & Cylinder Balancing:
   Firing order [1, 3, 4, 2]. Bank 1 = {1, 4}, Bank 2 = {2, 3}.
   Evaluates Gate 4 threshold: max_k |Delta omega_k| < 0.8 rad/s^2.
   Computes 0.5-order, 1.0-order, and 2.0-order harmonic components.

3. Deterministic 8-Fault Disambiguation Engine:
   Evaluates the Jacobian rank gate and deterministically isolates all 8 fault modes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml


# =============================================================================
# 1. Hardware Architecture Constants & Cylinder-to-Bank Mappings
# =============================================================================

FIRING_ORDER: Tuple[int, ...] = (1, 3, 4, 2)
BANK_1_CYLINDERS: Tuple[int, ...] = (1, 4)
BANK_2_CYLINDERS: Tuple[int, ...] = (2, 3)

CYLINDER_TO_BANK_MAP: Dict[int, int] = {
    1: 1,
    2: 2,
    3: 2,
    4: 1,
}

BANK_TO_CYLINDERS_MAP: Dict[int, Tuple[int, ...]] = {
    1: BANK_1_CYLINDERS,
    2: BANK_2_CYLINDERS,
}

VACUUM_VALVETRONIC_PA: float = 5000.0
VACUUM_THROTTLED_PA: float = 50000.0
LEAK_SCALE_THROTTLED: float = math.sqrt(10.0)
MICRO_LIFT_NOMINAL_MM: float = 0.35
HIGH_LIFT_THROTTLED_MM: float = 9.70

GATE1_MAX_FUEL_TRIM_FRAC: float = 0.035
GATE1_MAX_BANK_ASYM_FRAC: float = 0.020
GATE1_KOEO_HFM_V_MIN: float = 0.97
GATE1_KOEO_HFM_V_MAX: float = 1.03
GATE1_KOEO_MAX_MAF_KG_H: float = 0.5
GATE3_MAX_VANOS_ERROR_DEG: float = 0.8
GATE3_MIN_VANOS_SLEW_DEG_S: float = 80.0
GATE4_MAX_ROUGHNESS_RAD_S2: float = 0.8
JACOBIAN_RANK_GATE_COND_MAX: float = 50.0


class ExcitationRegime(str, Enum):
    """Canonical excitation regimes for active identifiability sequencing."""

    KOEO = "KOEO"
    VALVETRONIC_50MBAR = "VALVETRONIC_50MBAR"
    ACTIVE_THROTTLED_500MBAR = "ACTIVE_THROTTLED_500MBAR"
    COLD_WARMUP = "COLD_WARMUP"
    PART_LOAD_CRUISE = "PART_LOAD_CRUISE"
    CRUISE = "CRUISE"
    DYNAMIC_SLEW = "DYNAMIC_SLEW"


# =============================================================================
# 2. Canonical 8-Fault Mode Identifiers & Signatures
# =============================================================================

class FaultModeId(str, Enum):
    """The 8 canonical physical fault modes."""

    HEALTHY_BASELINE = "HEALTHY_BASELINE"
    FM_01_UNMETERED_AIR_CCV_LEAK = "FM_01_UNMETERED_AIR_CCV_LEAK"
    FM_02_HFM5_MAF_GAIN_DRIFT = "FM_02_HFM5_MAF_GAIN_DRIFT"
    FM_03_HFM5_MAF_ZERO_OFFSET = "FM_03_HFM5_MAF_ZERO_OFFSET"
    FM_04_FUEL_PRESSURE_REGULATOR_DROP = "FM_04_FUEL_PRESSURE_REGULATOR_DROP"
    FM_05_SINGLE_INJECTOR_FOULING = "FM_05_SINGLE_INJECTOR_FOULING"
    FM_06_VALVETRONIC_SHAFT_LEVER_WEAR = "FM_06_VALVETRONIC_SHAFT_LEVER_WEAR"
    FM_07_STICKY_VANOS_SOLENOID = "FM_07_STICKY_VANOS_SOLENOID"
    FM_08_THERMOSTAT_OR_RADIATOR_FAULT = "FM_08_THERMOSTAT_OR_RADIATOR_FAULT"


class ThermalSubFault(str, Enum):
    """Sub-modes for Fault #8 disambiguation."""

    THERMOSTAT_STUCK_OPEN = "THERMOSTAT_STUCK_OPEN"
    RADIATOR_FOULING_OR_CLOGGING = "RADIATOR_FOULING_OR_CLOGGING"


@dataclass(frozen=True)
class FaultSignatureSpec:
    """Physical signature specification for a single fault mode."""

    fault_id: int
    fault_code: FaultModeId
    name: str
    oem_part_numbers: List[str]
    sfi_code: str
    bank_symmetry: str
    primary_excitation: str
    r_maf_signature: str
    stoich_signature: str
    roughness_thermal_clue: str
    disambiguation_rule: str


CANONICAL_FAULT_SIGNATURES: Dict[FaultModeId, FaultSignatureSpec] = {
    FaultModeId.FM_01_UNMETERED_AIR_CCV_LEAK: FaultSignatureSpec(
        fault_id=1,
        fault_code=FaultModeId.FM_01_UNMETERED_AIR_CCV_LEAK,
        name="Unmetered Air / CCV Valve or Hose Leak",
        oem_part_numbers=["11617503520"],
        sfi_code="601.11",
        bank_symmetry="SYMMETRIC",
        primary_excitation="ACTIVE_THROTTLED_500MBAR",
        r_maf_signature="Constant negative offset: r_MAF = -m_dot_leak(dP_man) < 0",
        stoich_signature="Symmetric positive shift F_B1 ~= F_B2 proportional to +m_dot_leak/m_dot_air_calc",
        roughness_thermal_clue="Symmetric Delta omega_k",
        disambiguation_rule="50 -> 500 mbar switch scales m_dot_leak by sqrt(10) ~= 3.16x; KOEO V_HFM in [0.97, 1.03] V",
    ),
    FaultModeId.FM_02_HFM5_MAF_GAIN_DRIFT: FaultSignatureSpec(
        fault_id=2,
        fault_code=FaultModeId.FM_02_HFM5_MAF_GAIN_DRIFT,
        name="HFM5 MAF Gain Contamination / Aging",
        oem_part_numbers=["13627566986"],
        sfi_code="601.11",
        bank_symmetry="SYMMETRIC",
        primary_excitation="VALVETRONIC_50MBAR",
        r_maf_signature="Scales linearly with airflow: r_MAF(t) = theta_MAF,gain * m_dot_air_calc(t) < 0",
        stoich_signature="Symmetric load-invariant shift F_B1 ~= F_B2 ~= -theta_MAF,gain > 0",
        roughness_thermal_clue="Symmetric Delta omega_k",
        disambiguation_rule="Constant ratio r_MAF / m_dot_air_calc; invariant to 50 -> 500 mbar switch; KOEO normal",
    ),
    FaultModeId.FM_03_HFM5_MAF_ZERO_OFFSET: FaultSignatureSpec(
        fault_id=3,
        fault_code=FaultModeId.FM_03_HFM5_MAF_ZERO_OFFSET,
        name="HFM5 MAF Zero-Offset Drift",
        oem_part_numbers=["13627566986"],
        sfi_code="601.11",
        bank_symmetry="SYMMETRIC",
        primary_excitation="KOEO",
        r_maf_signature="Constant offset r_MAF = delta_MAF,offset != 0 across all airflows",
        stoich_signature="Symmetric hyperbolic decay F_B1 ~= F_B2 proportional to -delta_MAF,offset / m_dot_air_calc",
        roughness_thermal_clue="Symmetric Delta omega_k",
        disambiguation_rule="Stage 1 KOEO gate: at RPM=0, V_HFM not in [0.97 V, 1.03 V] or MAF > 0.5 kg/h",
    ),
    FaultModeId.FM_04_FUEL_PRESSURE_REGULATOR_DROP: FaultSignatureSpec(
        fault_id=4,
        fault_code=FaultModeId.FM_04_FUEL_PRESSURE_REGULATOR_DROP,
        name="Fuel Pressure Drop / Filter Regulator Degradation",
        oem_part_numbers=["13327512019", "11727545323"],
        sfi_code="601.21",
        bank_symmetry="SYMMETRIC",
        primary_excitation="VALVETRONIC_50MBAR",
        r_maf_signature="r_MAF(t) ~= 0 (measured airflow matches white-box physics)",
        stoich_signature="Symmetric positive divergence F_B1 ~= F_B2 > +0.08, rising with m_dot_fuel",
        roughness_thermal_clue="Symmetric lean surge at high load",
        disambiguation_rule="Decoupled from MAF gain error because r_MAF ~= 0 while F_total spikes with m_dot_fuel",
    ),
    FaultModeId.FM_05_SINGLE_INJECTOR_FOULING: FaultSignatureSpec(
        fault_id=5,
        fault_code=FaultModeId.FM_05_SINGLE_INJECTOR_FOULING,
        name="Single-Injector Varnish / Spray Fouling (Cyl 1-4)",
        oem_part_numbers=["13537546244"],
        sfi_code="601.21",
        bank_symmetry="ASYMMETRIC",
        primary_excitation="ACTIVE_THROTTLED_500MBAR",
        r_maf_signature="r_MAF(t) ~= 0",
        stoich_signature="Asymmetric bank split: affected bank F_total,Bx in [+0.03, +0.08]",
        roughness_thermal_clue="Asymmetric: Delta omega_k >> median(Delta omega) on fouled cylinder k",
        disambiguation_rule="Dual-bank trim asymmetry (|F_B1 - F_B2| > 0.03) persists in Throttled Mode (L_vvt = 9.7 mm)",
    ),
    FaultModeId.FM_06_VALVETRONIC_SHAFT_LEVER_WEAR: FaultSignatureSpec(
        fault_id=6,
        fault_code=FaultModeId.FM_06_VALVETRONIC_SHAFT_LEVER_WEAR,
        name="Valvetronic Eccentric Shaft / Intermediate Lever Wear",
        oem_part_numbers=["11377509295"],
        sfi_code="601.11",
        bank_symmetry="ASYMMETRIC",
        primary_excitation="ACTIVE_THROTTLED_500MBAR",
        r_maf_signature="Mild negative r_MAF at micro-lift (L_vvt < 0.6 mm); r_MAF -> 0 at L_vvt > 2.0 mm",
        stoich_signature="Mild bank asymmetry at 700 RPM idle that vanishes at higher valve lifts",
        roughness_thermal_clue="Asymmetric at micro-lift only: Delta omega_k > 0.8 rad/s^2 at 0.35 mm, vanishes at 9.7 mm",
        disambiguation_rule="Switching to Throttled Mode (L_vvt -> 9.7 mm) eliminates the cylinder roughness spike",
    ),
    FaultModeId.FM_07_STICKY_VANOS_SOLENOID: FaultSignatureSpec(
        fault_id=7,
        fault_code=FaultModeId.FM_07_STICKY_VANOS_SOLENOID,
        name="Sticky VANOS Solenoid / Cam Phasing Retard",
        oem_part_numbers=["11367560462"],
        sfi_code="601.11",
        bank_symmetry="SYMMETRIC",
        primary_excitation="VALVETRONIC_50MBAR",
        r_maf_signature="r_MAF < 0 during transient tip-in due to overlap error",
        stoich_signature="Transient STFT spike during cam slewing; normal steady-state LTFT",
        roughness_thermal_clue="Symmetric Delta omega_k; sluggish torque response",
        disambiguation_rule="Gate 3 violation: |theta_target - theta_actual| > 0.8 degCRK or slew < 80 degCRK/s at T_oil > 80 C",
    ),
    FaultModeId.FM_08_THERMOSTAT_OR_RADIATOR_FAULT: FaultSignatureSpec(
        fault_id=8,
        fault_code=FaultModeId.FM_08_THERMOSTAT_OR_RADIATOR_FAULT,
        name="Map-Thermostat Stuck Open vs. Radiator Fouling",
        oem_part_numbers=["11537510959"],
        sfi_code="601.41",
        bank_symmetry="NONE",
        primary_excitation="COLD_WARMUP",
        r_maf_signature="r_MAF ~= 0",
        stoich_signature="Normal (F_total ~= 0) once T_coolant > 78 C",
        roughness_thermal_clue="Thermal signature: T_rad,out rises prematurely or theta_rad_health < 0.65",
        disambiguation_rule="Stuck thermostat: T_rad,out tracks T_coolant with PWM = 0. Fouling: T_coolant > 108 C at PWM = 100%",
    ),
}

FAULT_PARAMETER_NAMES: Tuple[str, ...] = (
    "m_leak_ref",
    "theta_maf_gain",
    "delta_maf_offset",
    "k_fuel_press",
    "delta_inj_asym",
    "theta_vvt_wear",
    "theta_vanos_lag",
    "theta_thermal_fault",
)


# =============================================================================
# 3. Multi-Regime Observability Matrix & Sensitivity Jacobian
# =============================================================================

@dataclass
class JacobianRankGateResult:
    """Pre-excitation Jacobian rank and condition qualification result."""

    rank: int
    condition_number: float
    passed: bool
    requires_active_excitation: bool
    recommended_excitation: Optional[str]
    singular_values: List[float]
    regimes_evaluated: List[str]
    summary: str


def build_multi_regime_observability_matrix(
    regimes: Sequence[str] = (
        "KOEO",
        "VALVETRONIC_50MBAR",
        "ACTIVE_THROTTLED_500MBAR",
        "COLD_WARMUP",
    ),
    scale_500mbar: float = LEAK_SCALE_THROTTLED,
    c_air_cruise: float = 3.5,
    c_fuel_cruise: float = 3.5,
    c_vvt_microlift: float = 0.20,
    w_cruise_maf: float = 0.125,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Builds the stacked sensitivity Jacobian H and column-equilibrated H_tilde."""
    rows: List[List[float]] = []
    labels: List[str] = []
    regimes_set = {str(r.value if hasattr(r, "value") else r) for r in regimes}

    if "KOEO" in regimes_set:
        rows.append([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        labels.append("KOEO_MAF_residual")

    if "VALVETRONIC_50MBAR" in regimes_set or "VALVETRONIC" in regimes_set:
        rows.append([-1.0, 1.0, 1.0, 0.0, 0.0, -c_vvt_microlift, 0.0, 0.0])
        labels.append("VALVETRONIC_50MBAR_r_MAF")

        rows.append([1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        labels.append("VALVETRONIC_50MBAR_F_sym")

        rows.append([0.0, 0.0, 0.0, 0.0, 1.0, c_vvt_microlift, 0.0, 0.0])
        labels.append("VALVETRONIC_50MBAR_F_asym")

        rows.append([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        labels.append("VALVETRONIC_50MBAR_Delta_omega")

    if "ACTIVE_THROTTLED_500MBAR" in regimes_set or "THROTTLED" in regimes_set:
        rows.append([-scale_500mbar, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        labels.append("ACTIVE_THROTTLED_500MBAR_r_MAF")

        rows.append([scale_500mbar, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        labels.append("ACTIVE_THROTTLED_500MBAR_F_sym")

        rows.append([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        labels.append("ACTIVE_THROTTLED_500MBAR_Delta_omega")

    if "COLD_WARMUP" in regimes_set or "PART_LOAD_CRUISE" in regimes_set or "CRUISE" in regimes_set:
        rows.append([-w_cruise_maf / c_air_cruise, w_cruise_maf * c_air_cruise, w_cruise_maf, 0.0, 0.0, 0.0, 0.0, 0.0])
        labels.append("PART_LOAD_CRUISE_r_MAF")

        rows.append([1.0 / c_air_cruise, -1.0, -1.0 / c_air_cruise, c_fuel_cruise, 0.0, 0.0, 0.0, 0.0])
        labels.append("PART_LOAD_CRUISE_F_sym")

        rows.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        labels.append("DYNAMIC_SLEW_r_VANOS")

        rows.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        labels.append("COLD_WARMUP_r_Thermal")

    if not rows:
        return (
            np.zeros((0, 8), dtype=np.float64),
            np.zeros((0, 8), dtype=np.float64),
            [],
            list(FAULT_PARAMETER_NAMES),
        )

    H = np.array(rows, dtype=np.float64)
    col_norms = np.linalg.norm(H, axis=0)
    col_norms_safe = np.where(col_norms > 1e-12, col_norms, 1.0)
    H_tilde = H / col_norms_safe

    return H, H_tilde, labels, list(FAULT_PARAMETER_NAMES)


def evaluate_jacobian_rank_gate(
    regimes: Sequence[str] = (
        "KOEO",
        "VALVETRONIC_50MBAR",
        "ACTIVE_THROTTLED_500MBAR",
        "COLD_WARMUP",
    ),
    threshold: float = JACOBIAN_RANK_GATE_COND_MAX,
) -> JacobianRankGateResult:
    """Evaluates the Jacobian Rank Gate: cond(H_tilde) < 50.0 and rank == 8."""
    _, H_tilde, _, _ = build_multi_regime_observability_matrix(regimes=regimes)

    if H_tilde.shape[0] < 8:
        rank = int(np.linalg.matrix_rank(H_tilde)) if H_tilde.shape[0] > 0 else 0
        cond = float("inf")
    else:
        rank = int(np.linalg.matrix_rank(H_tilde))
        sv = np.linalg.svd(H_tilde, compute_uv=False)
        cond = float(sv[0] / sv[-1]) if sv[-1] > 1e-12 else float("inf")

    passed = (rank == 8) and (cond < threshold)
    requires_active = not passed

    regimes_set = set(regimes)
    recommended: Optional[str] = None
    if "KOEO" not in regimes_set:
        recommended = "KOEO"
    elif "ACTIVE_THROTTLED_500MBAR" not in regimes_set and "THROTTLED" not in regimes_set:
        recommended = "ACTIVE_THROTTLED_500MBAR"
    elif "COLD_WARMUP" not in regimes_set and "PART_LOAD_CRUISE" not in regimes_set:
        recommended = "PART_LOAD_CRUISE"
    elif "VALVETRONIC_50MBAR" not in regimes_set:
        recommended = "VALVETRONIC_50MBAR"

    sv_list = [float(s) for s in np.linalg.svd(H_tilde, compute_uv=False)] if H_tilde.shape[0] > 0 else []

    if passed:
        summary = (
            f"JACOBIAN RANK GATE PASS: Full column rank ({rank}/8) "
            f"with cond(H_tilde) = {cond:.3f} < {threshold:.1f}."
        )
    else:
        summary = (
            f"JACOBIAN RANK GATE FAIL: rank = {rank}/8, cond(H_tilde) = {cond:.3f} >= {threshold:.1f}. "
            f"Active excitation required: {recommended}."
        )

    return JacobianRankGateResult(
        rank=rank,
        condition_number=cond,
        passed=passed,
        requires_active_excitation=requires_active,
        recommended_excitation=recommended,
        singular_values=sv_list,
        regimes_evaluated=list(regimes),
        summary=summary,
    )


def numpy_multi_regime_residuals(
    theta: Sequence[float] | np.ndarray,
    scale_500mbar: float = LEAK_SCALE_THROTTLED,
    c_air_cruise: float = 3.5,
    c_fuel_cruise: float = 3.5,
    c_vvt_microlift: float = 0.20,
    w_cruise_maf: float = 0.125,
) -> np.ndarray:
    """Pure-NumPy forward residual map from 8 fault parameters theta to 12 multi-regime residuals."""
    th1, th2, th3, th4, th5, th6, th7, th8 = [float(x) for x in theta]

    return np.array(
        [
            th3,
            -th1 + th2 + th3 - c_vvt_microlift * th6,
            th1 - th2 - th3 + th4,
            th5 + c_vvt_microlift * th6,
            th5 + th6,
            -scale_500mbar * th1 + th2 + th3,
            scale_500mbar * th1 - th2 - th3 + th4,
            th5,
            (-w_cruise_maf / c_air_cruise) * th1 + (w_cruise_maf * c_air_cruise) * th2 + w_cruise_maf * th3,
            (1.0 / c_air_cruise) * th1 - th2 - (1.0 / c_air_cruise) * th3 + c_fuel_cruise * th4,
            th7,
            th8,
        ],
        dtype=np.float64,
    )


def finite_diff_observability_jacobian(
    theta: Optional[Sequence[float] | np.ndarray] = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Computes central finite-difference Jacobian of numpy_multi_regime_residuals."""
    theta_arr = np.zeros(8, dtype=np.float64) if theta is None else np.asarray(theta, dtype=np.float64)
    J = np.zeros((12, 8), dtype=np.float64)
    for j in range(8):
        e_j = np.zeros(8, dtype=np.float64)
        e_j[j] = eps
        f_plus = numpy_multi_regime_residuals(theta_arr + e_j)
        f_minus = numpy_multi_regime_residuals(theta_arr - e_j)
        J[:, j] = (f_plus - f_minus) / (2.0 * eps)
    return J


# =============================================================================
# 4. Crankshaft Angular Speed & Cylinder Balancing Analyzer
# =============================================================================

@dataclass
class CrankshaftRoughnessAnalysis:
    """Instantaneous crankshaft angular acceleration roughness and harmonic analysis."""

    rough_running: List[float]
    peak_cylinder: int
    suspect_bank: int
    max_spread_rad_s2: float
    gate4_passed: bool
    bank1_max_rad_s2: float
    bank2_max_rad_s2: float
    bank_asymmetry_rad_s2: float
    order_0_5_mag: float
    order_0_5_phase_rad: float
    order_1_0_mag: float
    order_1_0_phase_rad: float
    order_2_0_mag: float
    firing_order_roughness: List[float]
    sfi_code: str = "601.11"
    summary: str = ""


def analyze_crankshaft_roughness(
    rough_running: Sequence[float],
    rpm: float = 700.0,
) -> CrankshaftRoughnessAnalysis:
    """Performs cylinder-specific rough running evaluation and harmonic order decomposition."""
    if len(rough_running) != 4:
        raise ValueError(f"Expected 4 cylinder roughness values, got {len(rough_running)}")

    rr = [float(x) for x in rough_running]
    cyl1, cyl2, cyl3, cyl4 = rr
    y_firing = [cyl1, cyl3, cyl4, cyl2]

    abs_vals = [abs(x) for x in rr]
    max_idx = int(np.argmax(abs_vals))
    peak_cylinder = max_idx + 1
    max_spread = float(abs_vals[max_idx])

    bank1_max = float(max(abs(cyl1), abs(cyl4)))
    bank2_max = float(max(abs(cyl2), abs(cyl3)))
    suspect_bank = CYLINDER_TO_BANK_MAP[peak_cylinder]
    bank_asym = float(abs(bank1_max - bank2_max))
    gate4_passed = bool(max_spread < GATE4_MAX_ROUGHNESS_RAD_S2)

    a_05 = 0.5 * (y_firing[0] - y_firing[2])
    b_05 = 0.5 * (y_firing[1] - y_firing[3])
    mag_05 = float(math.sqrt(a_05 * a_05 + b_05 * b_05))
    phase_05 = float(math.atan2(b_05, a_05))

    a_10 = 0.5 * ((y_firing[0] + y_firing[2]) - (y_firing[1] + y_firing[3]))
    mag_10 = float(abs(a_10))
    phase_10 = 0.0 if a_10 >= 0 else float(math.pi)

    mean_val = float(np.mean(y_firing))
    mag_20 = float(math.sqrt(float(np.mean([(y - mean_val) ** 2 for y in y_firing]))))

    summary = (
        f"Gate 4 {'PASS' if gate4_passed else 'VIOLATION'}: max spread {max_spread:.3f} rad/s^2 "
        f"(peak Cyl {peak_cylinder}, Bank {suspect_bank}). "
        f"Harmonics: 0.5-order = {mag_05:.3f}, 1.0-order = {mag_10:.3f} rad/s^2."
    )

    return CrankshaftRoughnessAnalysis(
        rough_running=rr,
        peak_cylinder=peak_cylinder,
        suspect_bank=suspect_bank,
        max_spread_rad_s2=max_spread,
        gate4_passed=gate4_passed,
        bank1_max_rad_s2=bank1_max,
        bank2_max_rad_s2=bank2_max,
        bank_asymmetry_rad_s2=bank_asym,
        order_0_5_mag=mag_05,
        order_0_5_phase_rad=phase_05,
        order_1_0_mag=mag_10,
        order_1_0_phase_rad=phase_10,
        order_2_0_mag=mag_20,
        firing_order_roughness=y_firing,
        sfi_code="601.11",
        summary=summary,
    )


# =============================================================================
# 5. Multi-Regime Observations & Deterministic Isolation Engine
# =============================================================================

@dataclass
class MultiRegimeObservations:
    """Observations and diagnostic residuals recorded across excitation regimes."""

    koeo_hfm_voltage: Optional[float] = None
    koeo_maf_kg_h: Optional[float] = None

    valv_maf_residual_kg_h: Optional[float] = None
    valv_fuel_trim_b1: Optional[float] = None
    valv_fuel_trim_b2: Optional[float] = None
    valv_rough_running: Optional[Sequence[float]] = None
    valv_lift_mm: float = MICRO_LIFT_NOMINAL_MM

    throt_maf_residual_kg_h: Optional[float] = None
    throt_fuel_trim_b1: Optional[float] = None
    throt_fuel_trim_b2: Optional[float] = None
    throt_rough_running: Optional[Sequence[float]] = None
    throt_lift_mm: float = HIGH_LIFT_THROTTLED_MM

    cruise_maf_residual_kg_h: Optional[float] = None
    cruise_fuel_trim_b1: Optional[float] = None
    cruise_fuel_trim_b2: Optional[float] = None
    cruise_m_dot_fuel_kg_s: Optional[float] = None
    vanos_tracking_error_deg: Optional[float] = None
    vanos_slew_rate_deg_s: Optional[float] = None
    t_oil_c: float = 90.0
    t_coolant_c: Optional[float] = None
    t_rad_out_c: Optional[float] = None
    pwm_stat: Optional[float] = None
    theta_rad_health: Optional[float] = None


@dataclass
class IsolationResult:
    """Comprehensive diagnosis report produced by FaultIsolationEngine."""

    primary_fault: FaultModeId
    sub_fault: Optional[Union[str, ThermalSubFault]] = None
    confidence: float = 1.0
    requires_active_excitation: bool = False
    recommended_excitation: Optional[str] = None
    rank_gate_result: Optional[JacobianRankGateResult] = None
    oem_part_numbers: List[str] = field(default_factory=list)
    sfi_code: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""


class FaultIsolationEngine:
    """Deterministic 8-Fault Multi-Regime Physics Isolation & Disambiguation Engine."""

    def __init__(self, condition_threshold: float = JACOBIAN_RANK_GATE_COND_MAX) -> None:
        self.condition_threshold = float(condition_threshold)

    def identify_active_regimes(self, obs: MultiRegimeObservations) -> List[str]:
        """Identifies which excitation regimes are present in the provided observations."""
        regimes: List[str] = []

        if obs.koeo_hfm_voltage is not None or obs.koeo_maf_kg_h is not None:
            regimes.append("KOEO")

        if (
            obs.valv_maf_residual_kg_h is not None
            or obs.valv_fuel_trim_b1 is not None
            or obs.valv_rough_running is not None
        ):
            regimes.append("VALVETRONIC_50MBAR")

        if (
            obs.throt_maf_residual_kg_h is not None
            or obs.throt_fuel_trim_b1 is not None
            or obs.throt_rough_running is not None
        ):
            regimes.append("ACTIVE_THROTTLED_500MBAR")

        if (
            obs.cruise_maf_residual_kg_h is not None
            or obs.cruise_fuel_trim_b1 is not None
            or obs.vanos_tracking_error_deg is not None
            or obs.t_coolant_c is not None
            or obs.theta_rad_health is not None
        ):
            regimes.append("COLD_WARMUP")

        return regimes

    def isolate_faults(self, obs: MultiRegimeObservations) -> IsolationResult:
        """Deterministically isolates and disambiguates the active fault mode."""
        active_regimes = self.identify_active_regimes(obs)
        rank_result = evaluate_jacobian_rank_gate(
            regimes=active_regimes, threshold=self.condition_threshold
        )

        evidence: Dict[str, Any] = {
            "active_regimes": active_regimes,
            "rank_gate": {
                "rank": rank_result.rank,
                "cond": rank_result.condition_number,
                "passed": rank_result.passed,
            },
        }

        # Pre-Check: Fault #3 HFM5 Zero-Offset Drift (Gate 1 KOEO Violation)
        if obs.koeo_hfm_voltage is not None or obs.koeo_maf_kg_h is not None:
            v_koeo = obs.koeo_hfm_voltage if obs.koeo_hfm_voltage is not None else 1.00
            maf_koeo = obs.koeo_maf_kg_h if obs.koeo_maf_kg_h is not None else 0.0

            v_viol = (v_koeo < GATE1_KOEO_HFM_V_MIN) or (v_koeo > GATE1_KOEO_HFM_V_MAX)
            maf_viol = maf_koeo > GATE1_KOEO_MAX_MAF_KG_H

            if v_viol or maf_viol:
                spec = CANONICAL_FAULT_SIGNATURES[FaultModeId.FM_03_HFM5_MAF_ZERO_OFFSET]
                evidence["koeo_voltage_v"] = v_koeo
                evidence["koeo_maf_kg_h"] = maf_koeo
                summary = (
                    f"Fault #3 ISOLATED: HFM5 Zero-Offset Drift. "
                    f"KOEO voltage V_HFM = {v_koeo:.3f} V (bounds [0.97, 1.03]) or "
                    f"MAF = {maf_koeo:.2f} kg/h > 0.5 kg/h."
                )
                return IsolationResult(
                    primary_fault=FaultModeId.FM_03_HFM5_MAF_ZERO_OFFSET,
                    confidence=1.0,
                    requires_active_excitation=False,
                    rank_gate_result=rank_result,
                    oem_part_numbers=spec.oem_part_numbers,
                    sfi_code=spec.sfi_code,
                    evidence=evidence,
                    summary=summary,
                )

        # Pre-Check: Fault #7 Sticky VANOS Solenoid (Gate 3 Tracking Violation)
        if obs.vanos_tracking_error_deg is not None or obs.vanos_slew_rate_deg_s is not None:
            err_deg = obs.vanos_tracking_error_deg if obs.vanos_tracking_error_deg is not None else 0.0
            slew_rate = obs.vanos_slew_rate_deg_s if obs.vanos_slew_rate_deg_s is not None else 100.0

            is_warm = obs.t_oil_c >= 80.0
            tracking_viol = err_deg > GATE3_MAX_VANOS_ERROR_DEG
            slew_viol = is_warm and (slew_rate < GATE3_MIN_VANOS_SLEW_DEG_S)

            if tracking_viol or slew_viol:
                spec = CANONICAL_FAULT_SIGNATURES[FaultModeId.FM_07_STICKY_VANOS_SOLENOID]
                evidence["vanos_error_deg"] = err_deg
                evidence["vanos_slew_deg_s"] = slew_rate
                summary = (
                    f"Fault #7 ISOLATED: Sticky VANOS Solenoid / Cam Phasing Retard. "
                    f"Tracking error = {err_deg:.2f} degCRK (> 0.8) or slew rate = {slew_rate:.1f} deg/s (< 80 at T_oil > 80 C)."
                )
                return IsolationResult(
                    primary_fault=FaultModeId.FM_07_STICKY_VANOS_SOLENOID,
                    confidence=1.0,
                    requires_active_excitation=False,
                    rank_gate_result=rank_result,
                    oem_part_numbers=spec.oem_part_numbers,
                    sfi_code=spec.sfi_code,
                    evidence=evidence,
                    summary=summary,
                )

        # Pre-Check: Fault #8 Map-Thermostat Stuck Open vs Radiator Fouling
        if (
            obs.t_coolant_c is not None
            or obs.t_rad_out_c is not None
            or obs.theta_rad_health is not None
        ):
            t_cool = obs.t_coolant_c if obs.t_coolant_c is not None else 90.0
            t_rad = obs.t_rad_out_c if obs.t_rad_out_c is not None else 40.0
            pwm = obs.pwm_stat if obs.pwm_stat is not None else 0.0
            rad_health = obs.theta_rad_health if obs.theta_rad_health is not None else 1.0

            stuck_open = (t_cool < 85.0) and (pwm < 0.05) and (t_rad > 50.0 and (t_cool - t_rad) < 15.0)
            rad_clogged = (t_cool > 108.0 and pwm > 0.80) or (rad_health < 0.65)

            if stuck_open or rad_clogged:
                spec = CANONICAL_FAULT_SIGNATURES[FaultModeId.FM_08_THERMOSTAT_OR_RADIATOR_FAULT]
                sub = (
                    ThermalSubFault.THERMOSTAT_STUCK_OPEN
                    if stuck_open
                    else ThermalSubFault.RADIATOR_FOULING_OR_CLOGGING
                )
                evidence["t_coolant_c"] = t_cool
                evidence["t_rad_out_c"] = t_rad
                evidence["pwm_stat"] = pwm
                evidence["theta_rad_health"] = rad_health
                evidence["thermal_sub_fault"] = sub.value
                summary = (
                    f"Fault #8 ISOLATED: Map-Thermostat Stuck Open vs Radiator Fouling ({sub.value}). "
                    f"T_coolant={t_cool:.1f} C, T_rad_out={t_rad:.1f} C, PWM={pwm*100:.0f}%, theta_rad={rad_health:.2f}."
                )
                return IsolationResult(
                    primary_fault=FaultModeId.FM_08_THERMOSTAT_OR_RADIATOR_FAULT,
                    sub_fault=sub,
                    confidence=1.0,
                    requires_active_excitation=False,
                    rank_gate_result=rank_result,
                    oem_part_numbers=spec.oem_part_numbers,
                    sfi_code=spec.sfi_code,
                    evidence=evidence,
                    summary=summary,
                )

        # Asymmetric Roughness & Stoichiometric Split (Fault #5 vs Fault #6)
        valv_rr_analysis: Optional[CrankshaftRoughnessAnalysis] = None
        if obs.valv_rough_running is not None:
            valv_rr_analysis = analyze_crankshaft_roughness(obs.valv_rough_running)
            evidence["valv_roughness"] = {
                "max_spread": valv_rr_analysis.max_spread_rad_s2,
                "peak_cylinder": valv_rr_analysis.peak_cylinder,
                "suspect_bank": valv_rr_analysis.suspect_bank,
                "gate4_passed": valv_rr_analysis.gate4_passed,
                "order_0_5": valv_rr_analysis.order_0_5_mag,
                "order_1_0": valv_rr_analysis.order_1_0_mag,
            }

        throt_rr_analysis: Optional[CrankshaftRoughnessAnalysis] = None
        if obs.throt_rough_running is not None:
            throt_rr_analysis = analyze_crankshaft_roughness(obs.throt_rough_running)
            evidence["throt_roughness"] = {
                "max_spread": throt_rr_analysis.max_spread_rad_s2,
                "peak_cylinder": throt_rr_analysis.peak_cylinder,
                "suspect_bank": throt_rr_analysis.suspect_bank,
                "gate4_passed": throt_rr_analysis.gate4_passed,
            }

        valv_asym = 0.0
        if obs.valv_fuel_trim_b1 is not None and obs.valv_fuel_trim_b2 is not None:
            valv_asym = abs(obs.valv_fuel_trim_b1 - obs.valv_fuel_trim_b2)
            evidence["valv_trim_asym"] = valv_asym

        if valv_rr_analysis is not None and not valv_rr_analysis.gate4_passed:
            if throt_rr_analysis is None:
                return IsolationResult(
                    primary_fault=FaultModeId.FM_05_SINGLE_INJECTOR_FOULING,
                    confidence=0.5,
                    requires_active_excitation=True,
                    recommended_excitation="ACTIVE_THROTTLED_500MBAR",
                    rank_gate_result=rank_result,
                    evidence=evidence,
                    summary=(
                        f"AMBIGUOUS CYLINDER ROUGHNESS: Peak Cyl {valv_rr_analysis.peak_cylinder} "
                        f"({valv_rr_analysis.max_spread_rad_s2:.2f} rad/s^2). "
                        "Active Throttled excitation (L_vvt = 9.7 mm) required to separate "
                        "Injector Fouling (#5) from Valvetronic Wear (#6)."
                    ),
                )

            roughness_vanished = throt_rr_analysis.gate4_passed
            asym_vanished = valv_asym < GATE1_MAX_BANK_ASYM_FRAC

            if roughness_vanished and asym_vanished:
                spec = CANONICAL_FAULT_SIGNATURES[FaultModeId.FM_06_VALVETRONIC_SHAFT_LEVER_WEAR]
                summary = (
                    f"Fault #6 ISOLATED: Valvetronic Intermediate Lever / Shaft Wear. "
                    f"Micro-lift roughness {valv_rr_analysis.max_spread_rad_s2:.2f} rad/s^2 "
                    f"vanished to {throt_rr_analysis.max_spread_rad_s2:.2f} rad/s^2 in Active Throttled Mode (9.7 mm)."
                )
                return IsolationResult(
                    primary_fault=FaultModeId.FM_06_VALVETRONIC_SHAFT_LEVER_WEAR,
                    sub_fault=f"CYL_{valv_rr_analysis.peak_cylinder}",
                    confidence=1.0,
                    requires_active_excitation=False,
                    rank_gate_result=rank_result,
                    oem_part_numbers=spec.oem_part_numbers,
                    sfi_code=spec.sfi_code,
                    evidence=evidence,
                    summary=summary,
                )
            else:
                spec = CANONICAL_FAULT_SIGNATURES[FaultModeId.FM_05_SINGLE_INJECTOR_FOULING]
                sub = f"BANK{valv_rr_analysis.suspect_bank}_CYL{valv_rr_analysis.peak_cylinder}"
                summary = (
                    f"Fault #5 ISOLATED: Single-Injector Varnish / Spray Fouling ({sub}). "
                    f"Asymmetry and roughness persisted in Throttled Mode "
                    f"({throt_rr_analysis.max_spread_rad_s2:.2f} rad/s^2, bank split {valv_asym*100:.1f}%)."
                )
                return IsolationResult(
                    primary_fault=FaultModeId.FM_05_SINGLE_INJECTOR_FOULING,
                    sub_fault=sub,
                    confidence=1.0,
                    requires_active_excitation=False,
                    rank_gate_result=rank_result,
                    oem_part_numbers=spec.oem_part_numbers,
                    sfi_code=spec.sfi_code,
                    evidence=evidence,
                    summary=summary,
                )

        # Symmetric Airflow & Fuel Trim Anomaly (Fault #1, #2, #4 vs Healthy)
        valv_maf = obs.valv_maf_residual_kg_h if obs.valv_maf_residual_kg_h is not None else 0.0
        valv_trim_b1 = obs.valv_fuel_trim_b1 if obs.valv_fuel_trim_b1 is not None else 0.0
        valv_trim_b2 = obs.valv_fuel_trim_b2 if obs.valv_fuel_trim_b2 is not None else 0.0
        valv_trim_sym = 0.5 * (valv_trim_b1 + valv_trim_b2)

        cruise_trim_b1 = obs.cruise_fuel_trim_b1 if obs.cruise_fuel_trim_b1 is not None else valv_trim_b1
        cruise_trim_b2 = obs.cruise_fuel_trim_b2 if obs.cruise_fuel_trim_b2 is not None else valv_trim_b2
        cruise_trim_sym = 0.5 * (cruise_trim_b1 + cruise_trim_b2)

        evidence["valv_maf_residual_kg_h"] = valv_maf
        evidence["valv_trim_sym"] = valv_trim_sym
        evidence["cruise_trim_sym"] = cruise_trim_sym

        is_healthy = (
            abs(valv_trim_sym) < GATE1_MAX_FUEL_TRIM_FRAC
            and valv_asym < GATE1_MAX_BANK_ASYM_FRAC
            and abs(valv_maf) < 1.0
        )

        if is_healthy and (valv_rr_analysis is None or valv_rr_analysis.gate4_passed):
            return IsolationResult(
                primary_fault=FaultModeId.HEALTHY_BASELINE,
                confidence=1.0,
                requires_active_excitation=False,
                rank_gate_result=rank_result,
                evidence=evidence,
                summary="HEALTHY BASELINE: All breathing, stoichiometric, roughness, and thermal signals within golden envelope.",
            )

        if abs(valv_maf) < 1.0 and (valv_trim_sym > 0.05 or cruise_trim_sym > 0.08):
            spec = CANONICAL_FAULT_SIGNATURES[FaultModeId.FM_04_FUEL_PRESSURE_REGULATOR_DROP]
            summary = (
                f"Fault #4 ISOLATED: Fuel Pressure Drop / FPR Filter Degradation. "
                f"Airflow matches physics (r_MAF = {valv_maf:.2f} kg/h), while fuel trims "
                f"spike lean with fuel demand (F_sym = {cruise_trim_sym*100:.1f}%)."
            )
            return IsolationResult(
                primary_fault=FaultModeId.FM_04_FUEL_PRESSURE_REGULATOR_DROP,
                confidence=1.0,
                requires_active_excitation=False,
                rank_gate_result=rank_result,
                oem_part_numbers=spec.oem_part_numbers,
                sfi_code=spec.sfi_code,
                evidence=evidence,
                summary=summary,
            )

        if valv_maf < -1.0 or valv_trim_sym > GATE1_MAX_FUEL_TRIM_FRAC:
            if obs.throt_maf_residual_kg_h is None and obs.throt_fuel_trim_b1 is None:
                return IsolationResult(
                    primary_fault=FaultModeId.FM_01_UNMETERED_AIR_CCV_LEAK,
                    confidence=0.5,
                    requires_active_excitation=True,
                    recommended_excitation="ACTIVE_THROTTLED_500MBAR",
                    rank_gate_result=rank_result,
                    evidence=evidence,
                    summary=(
                        f"AMBIGUOUS LEAK / MAF GAIN: r_MAF = {valv_maf:.2f} kg/h, F_sym = {valv_trim_sym*100:.1f}%. "
                        "Active Throttled excitation (500 mbar vacuum switch) required to test for "
                        "sqrt(10) ~= 3.16x leak scaling."
                    ),
                )

            throt_maf = obs.throt_maf_residual_kg_h if obs.throt_maf_residual_kg_h is not None else 0.0
            leak_ratio = abs(throt_maf) / abs(valv_maf) if abs(valv_maf) > 0.1 else 1.0
            evidence["leak_ratio_500_to_50"] = leak_ratio

            if leak_ratio >= 2.2:
                spec = CANONICAL_FAULT_SIGNATURES[FaultModeId.FM_01_UNMETERED_AIR_CCV_LEAK]
                summary = (
                    f"Fault #1 ISOLATED: Unmetered Air / CCV Valve or Hose Leak. "
                    f"Leak flow scaled by {leak_ratio:.2f}x (nominal sqrt(10) ~= 3.16x) under 500 mbar Active Throttled mode."
                )
                return IsolationResult(
                    primary_fault=FaultModeId.FM_01_UNMETERED_AIR_CCV_LEAK,
                    confidence=1.0,
                    requires_active_excitation=False,
                    rank_gate_result=rank_result,
                    oem_part_numbers=spec.oem_part_numbers,
                    sfi_code=spec.sfi_code,
                    evidence=evidence,
                    summary=summary,
                )
            else:
                spec = CANONICAL_FAULT_SIGNATURES[FaultModeId.FM_02_HFM5_MAF_GAIN_DRIFT]
                summary = (
                    f"Fault #2 ISOLATED: HFM5 MAF Gain Contamination / Aging. "
                    f"Airflow residual ratio under vacuum switch was {leak_ratio:.2f}x (load-invariant, not vacuum-scaled)."
                )
                return IsolationResult(
                    primary_fault=FaultModeId.FM_02_HFM5_MAF_GAIN_DRIFT,
                    confidence=1.0,
                    requires_active_excitation=False,
                    rank_gate_result=rank_result,
                    oem_part_numbers=spec.oem_part_numbers,
                    sfi_code=spec.sfi_code,
                    evidence=evidence,
                    summary=summary,
                )

        return IsolationResult(
            primary_fault=FaultModeId.HEALTHY_BASELINE,
            confidence=0.8,
            requires_active_excitation=rank_result.requires_active_excitation,
            recommended_excitation=rank_result.recommended_excitation,
            rank_gate_result=rank_result,
            evidence=evidence,
            summary="HEALTHY BASELINE: No active fault modes matched observations.",
        )


def isolate_faults(
    observations: MultiRegimeObservations,
    condition_threshold: float = JACOBIAN_RANK_GATE_COND_MAX,
) -> IsolationResult:
    """Convenience functional wrapper for FaultIsolationEngine.isolate_faults."""
    return FaultIsolationEngine(condition_threshold=condition_threshold).isolate_faults(observations)


def load_fault_isolation_matrix(
    path_or_str: Union[str, Path] = "fault_isolation_matrix.yaml",
) -> Dict[str, Any]:
    """Loads the 8-fault isolation matrix YAML from repo config/ or bundled package data."""
    from greybox_showcase.iso19848_twin import resolve_config_path

    resolved = resolve_config_path(path_or_str)
    with open(resolved, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Invalid YAML content in {resolved}")

    return raw
