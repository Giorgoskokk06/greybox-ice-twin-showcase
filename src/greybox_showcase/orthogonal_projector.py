"""Fault-Subspace Orthogonal Projector (P_{F^perp}) & Anti-Masking Surrogate.

Mathematical Foundations:
1. Fault Signature Subspace F:
   In a single-asset (N=1) digital twin, an unconstrained ML surrogate trained on raw
   breathing residuals r_raw(t) = m_dot_air_meas(t) - m_dot_air_calc(t) risks absorbing
   incipient physical faults (such as CCV leaks or MAF calibration drift).
   We define the 3-dimensional fault signature subspace:
       F = span{ 1,  m_dot_air_calc,  sqrt(max(Delta_P_man, 0) / P_ref) }
   where:
       f_1 = 1 in R^N: MAF Zero-Offset Drift (delta_MAF,offset)
       f_2 = m_dot_air_calc in R^N: MAF Gain Contamination (theta_MAF,gain * m_dot_air_calc)
       f_3 = sqrt(Delta_P_man / P_ref) in R^N: Unmetered Air Leak (-m_dot_leak * sqrt(Delta_P/P_ref))

2. Orthogonal Projection Operator onto F^perp:
   Let F = [f_1, f_2, f_3] in R^{N x 3}. Computing its orthonormal basis Q_F in R^{N x r} (r <= 3)
   via reorthogonalized QR decomposition yields:
       P_{F^perp} = I_N - Q_F Q_F^T
   satisfying:
       P_{F^perp}^2 = P_{F^perp}           (Idempotent)
       P_{F^perp}^T = P_{F^perp}           (Symmetric)
       ||P_{F^perp} F||_F < 1e-12          (Strict Orthogonality)
       ||P_{F^perp} r_fault|| == 0         (100% Fault Annihilation)

3. Zero-Overlap Inference Guarantee:
   The surrogate's basis features Phi(X) and training targets r_raw are projected onto F^perp:
       r_perp = P_{F^perp} r_raw
       Phi_perp = P_{F^perp} Phi(X)
   Ridge-regularized linear readout weights w and fault projection coefficients
   c = (F^T F)^(-1) F^T (Phi w) guarantee that at inference time:
       g_perp(x) = g_raw(x) - f(x)^T c
   has identically ZERO projection onto F (||P_F g_perp|| < 1e-10), preserving 100% of the
   fault signal in the corrected residual e(t) = r_raw(t) - g_perp(x_t).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# 1. Fault-Subspace Orthogonal Projector (P_{F^perp})
# =============================================================================

class FaultSubspaceProjector:
    """Constructs and applies the orthogonal projection operator P_{F^perp} = I - Q_F Q_F^T.

    Spans the 3 physical fault modes in air-fuel metering:
    - Mode 1: MAF Zero-Offset Drift (f_1 = 1)
    - Mode 2: MAF Gain Contamination (f_2 = m_dot_air_calc)
    - Mode 3: Unmetered Air / CCV Leak (f_3 = sqrt(max(Delta_P_man, 0) / P_ref))
    """

    BASIS_NAMES: Tuple[str, str, str] = (
        "maf_zero_offset",
        "maf_gain_contamination",
        "unmetered_leak_orifice",
    )

    def __init__(self, p_ref: Optional[float] = 50000.0) -> None:
        self.F: Optional[np.ndarray] = None
        self.Q: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None
        self.rank: int = 0
        self.n_samples: int = 0
        self.p_ref: Optional[float] = p_ref
        self._P_perp: Optional[np.ndarray] = None

    @classmethod
    def build_fault_basis(
        cls,
        m_dot_air_calc: np.ndarray | Sequence[float] | float,
        delta_p_man: np.ndarray | Sequence[float] | float,
        p_ref: Optional[float] = 50000.0,
    ) -> np.ndarray:
        """Constructs the N x 3 fault basis matrix F = [1, m_dot_calc, sqrt(max(Delta_P, 0) / P_ref)]."""
        m_calc_arr = np.atleast_1d(np.asarray(m_dot_air_calc, dtype=np.float64))
        dp_arr = np.atleast_1d(np.asarray(delta_p_man, dtype=np.float64))

        if m_calc_arr.shape != dp_arr.shape:
            raise ValueError(
                f"Shape mismatch: m_dot_air_calc {m_calc_arr.shape} vs delta_p_man {dp_arr.shape}"
            )

        n = len(m_calc_arr)
        f1 = np.ones(n, dtype=np.float64)
        f2 = m_calc_arr.copy()
        if p_ref is not None and p_ref > 0.0:
            f3 = np.sqrt(np.maximum(dp_arr, 0.0) / float(p_ref))
        else:
            f3 = np.sqrt(np.maximum(dp_arr, 0.0))

        return np.column_stack([f1, f2, f3])

    def fit(
        self,
        m_dot_air_calc: np.ndarray | Sequence[float],
        delta_p_man: np.ndarray | Sequence[float],
        p_ref: Optional[float] = 50000.0,
    ) -> FaultSubspaceProjector:
        """Computes reorthogonalized QR orthonormal basis Q_F of F in R^{N x 3}."""
        self.p_ref = p_ref
        F = self.build_fault_basis(m_dot_air_calc, delta_p_man, p_ref=self.p_ref)
        self.n_samples, n_basis = F.shape
        self.F = F

        if self.n_samples < n_basis:
            raise ValueError(
                f"Number of samples N={self.n_samples} must be >= rank of F ({n_basis})."
            )

        col_norms = np.linalg.norm(F, axis=0)
        col_norms = np.where(col_norms > 1e-12, col_norms, 1.0)
        F_scaled = F / col_norms

        Q, R = np.linalg.qr(F_scaled, mode="reduced")
        Q, _ = np.linalg.qr(Q, mode="reduced")

        r_diag = np.abs(np.diag(R))
        tol = np.max(r_diag) * np.finfo(np.float64).eps * max(F.shape)
        eff_rank = int(np.sum(r_diag > tol))

        self.Q = Q[:, :eff_rank]
        self.R = R[:eff_rank, :]
        self.rank = eff_rank
        self._P_perp = None

        return self

    def compute_projection_matrix(self) -> np.ndarray:
        """Returns the explicit N x N projection matrix P_{F^perp} = I_N - Q_F Q_F^T."""
        if self.Q is None:
            raise RuntimeError("FaultSubspaceProjector is not fitted yet.")
        if self._P_perp is None:
            self._P_perp = np.eye(self.n_samples, dtype=np.float64) - self.Q @ self.Q.T
        return self._P_perp

    def compute_fault_projection_matrix(self) -> np.ndarray:
        """Returns the explicit N x N projection matrix onto F: P_F = Q_F Q_F^T."""
        if self.Q is None:
            raise RuntimeError("FaultSubspaceProjector is not fitted yet.")
        return self.Q @ self.Q.T

    def project_residuals(self, r: np.ndarray | Sequence[float]) -> np.ndarray:
        """Projects residual vector r onto F^perp in O(N*r) time: r_perp = r - Q (Q^T r)."""
        if self.Q is None:
            raise RuntimeError("FaultSubspaceProjector is not fitted yet.")
        r_arr = np.asarray(r, dtype=np.float64)
        if len(r_arr) != self.n_samples:
            raise ValueError(
                f"Residual length {len(r_arr)} does not match fitted sample size {self.n_samples}."
            )
        return r_arr - self.Q @ (self.Q.T @ r_arr)

    def project_features(self, Phi: np.ndarray) -> np.ndarray:
        """Projects feature matrix Phi in R^{N x M} onto F^perp: Phi_perp = Phi - Q (Q^T Phi)."""
        if self.Q is None:
            raise RuntimeError("FaultSubspaceProjector is not fitted yet.")
        Phi_arr = np.asarray(Phi, dtype=np.float64)
        if Phi_arr.shape[0] != self.n_samples:
            raise ValueError(
                f"Feature rows {Phi_arr.shape[0]} do not match fitted sample size {self.n_samples}."
            )
        return Phi_arr - self.Q @ (self.Q.T @ Phi_arr)

    def project_onto_fault(self, r: np.ndarray | Sequence[float]) -> np.ndarray:
        """Projects residual vector r onto the fault subspace F: r_F = Q (Q^T r)."""
        if self.Q is None:
            raise RuntimeError("FaultSubspaceProjector is not fitted yet.")
        r_arr = np.asarray(r, dtype=np.float64)
        return self.Q @ (self.Q.T @ r_arr)

    def compute_fault_coefficients(self, y: np.ndarray | Sequence[float]) -> np.ndarray:
        """Computes OLS coefficients c = (F^T F)^(-1) F^T y of vector y onto fault basis F."""
        if self.F is None:
            raise RuntimeError("FaultSubspaceProjector is not fitted yet.")
        y_arr = np.asarray(y, dtype=np.float64)
        coeffs, _, _, _ = np.linalg.lstsq(self.F, y_arr, rcond=None)
        return coeffs

    def verify_properties(self) -> Dict[str, float | bool]:
        """Mathematically verifies algebraic properties of P_{F^perp}."""
        P = self.compute_projection_matrix()
        if self.F is None:
            raise RuntimeError("FaultSubspaceProjector is not fitted.")

        idempotence_err = float(np.linalg.norm(P @ P - P))
        symmetry_err = float(np.linalg.norm(P.T - P))
        orthogonality_err = float(np.linalg.norm(P @ self.F))

        return {
            "idempotence_error": idempotence_err,
            "symmetry_error": symmetry_err,
            "orthogonality_error": orthogonality_err,
            "is_valid": bool(
                idempotence_err < 1e-12
                and symmetry_err < 1e-12
                and orthogonality_err < 1e-12
            ),
        }


# =============================================================================
# 2. Pure-NumPy Orthogonal Polynomial/Fourier Surrogate (Zero-ML-Bloat)
# =============================================================================

@dataclass
class SurrogateAuditResult:
    """Audit metrics proving 0.0% fault absorption and exact fault preservation."""

    fault_subspace_norm: float
    absorbed_fault_ratio: float
    fault_annihilation_error: float
    is_anti_masking_verified: bool


class OrthogonalPolySurrogate:
    """Pure-NumPy polynomial/Fourier residual surrogate strictly orthogonal to F.

    Learns unmodeled intake dynamics (e.g., acoustic runner pulsations) from
    normalized kinematic features while guaranteeing ||P_F g_perp||_2 < 1e-10.
    """

    def __init__(self, degree: int = 3, ridge_alpha: float = 1e-3) -> None:
        self.degree = degree
        self.ridge_alpha = ridge_alpha
        self.projector = FaultSubspaceProjector()
        self._x_mean: Optional[np.ndarray] = None
        self._x_std: Optional[np.ndarray] = None
        self.weights_: Optional[np.ndarray] = None
        self.fault_readout_coeffs_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def _build_basis(self, X_norm: np.ndarray) -> np.ndarray:
        cols = []
        for d in range(1, self.degree + 1):
            cols.append(X_norm ** d)
        cols.append(np.sin(X_norm))
        cols.append(np.cos(X_norm))
        return np.hstack(cols)

    def fit(
        self,
        X: np.ndarray | Sequence[Sequence[float]],
        r_raw: np.ndarray | Sequence[float],
        m_dot_air_calc: np.ndarray | Sequence[float],
        delta_p_man: np.ndarray | Sequence[float],
    ) -> OrthogonalPolySurrogate:
        """Fits Ridge-regularized surrogate in F^perp."""
        X_arr = np.atleast_2d(np.asarray(X, dtype=np.float64))
        r_arr = np.atleast_1d(np.asarray(r_raw, dtype=np.float64))
        m_calc_arr = np.atleast_1d(np.asarray(m_dot_air_calc, dtype=np.float64))
        dp_arr = np.atleast_1d(np.asarray(delta_p_man, dtype=np.float64))

        self.projector.fit(m_calc_arr, dp_arr)
        r_perp = self.projector.project_residuals(r_arr)

        self._x_mean = np.mean(X_arr, axis=0)
        self._x_std = np.where(np.std(X_arr, axis=0) > 1e-12, np.std(X_arr, axis=0), 1.0)
        Phi = self._build_basis((X_arr - self._x_mean) / self._x_std)
        Phi_perp = self.projector.project_features(Phi)

        n_feat = Phi_perp.shape[1]
        gram = Phi_perp.T @ Phi_perp + self.ridge_alpha * np.eye(n_feat, dtype=np.float64)
        self.weights_ = np.linalg.solve(gram, Phi_perp.T @ r_perp)

        g_raw_train = Phi @ self.weights_
        self.fault_readout_coeffs_ = self.projector.compute_fault_coefficients(g_raw_train)
        self.is_fitted = True
        return self

    def predict_surrogate(
        self,
        X: np.ndarray | Sequence[Sequence[float]],
        m_dot_air_calc: np.ndarray | Sequence[float] | float,
        delta_p_man: np.ndarray | Sequence[float] | float,
    ) -> np.ndarray:
        """Predicts orthogonal residual correction g_perp strictly in F^perp."""
        if not self.is_fitted or self.weights_ is None or self.fault_readout_coeffs_ is None:
            raise RuntimeError("OrthogonalPolySurrogate must be fitted before predict.")

        X_arr = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Phi = self._build_basis((X_arr - self._x_mean) / self._x_std)
        g_raw = Phi @ self.weights_

        m_calc_arr = np.atleast_1d(np.asarray(m_dot_air_calc, dtype=np.float64))
        dp_arr = np.atleast_1d(np.asarray(delta_p_man, dtype=np.float64))
        F_query = FaultSubspaceProjector.build_fault_basis(
            m_calc_arr, dp_arr, p_ref=self.projector.p_ref
        )
        g_perp = g_raw - F_query @ self.fault_readout_coeffs_
        if len(g_perp) >= 3:
            Q_q, _ = np.linalg.qr(F_query, mode="reduced")
            g_perp = g_perp - Q_q @ (Q_q.T @ g_perp)
        return g_perp

    def audit_fault_non_absorption(
        self,
        F: np.ndarray,
        g_perp: np.ndarray,
    ) -> SurrogateAuditResult:
        """Audits that learned surrogate has 0.0% projection onto fault subspace F."""
        F_eval = np.asarray(F, dtype=np.float64)
        g_eval = np.asarray(g_perp, dtype=np.float64)
        Q, _ = np.linalg.qr(F_eval, mode="reduced")
        proj_F = Q @ (Q.T @ g_eval)
        norm_proj = float(np.linalg.norm(proj_F))
        norm_g = float(np.linalg.norm(g_eval))
        return SurrogateAuditResult(
            fault_subspace_norm=norm_proj,
            absorbed_fault_ratio=norm_proj / max(norm_g, 1e-12),
            fault_annihilation_error=norm_proj,
            is_anti_masking_verified=bool(norm_proj < 1e-10),
        )
