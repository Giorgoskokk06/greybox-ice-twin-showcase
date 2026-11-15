"""Unit tests for FaultSubspaceProjector and OrthogonalPolySurrogate."""

from __future__ import annotations

import numpy as np
import pytest

from greybox_showcase.orthogonal_projector import (
    FaultSubspaceProjector,
    OrthogonalPolySurrogate,
)


class TestFaultSubspaceProjector:
    """Verifies algebraic properties of P_{F^perp} = I - Q_F Q_F^T."""

    def test_projector_algebraic_properties_and_annihilation(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        m_calc = rng.uniform(0.003, 0.05, n)
        dp_man = rng.uniform(4000.0, 50000.0, n)

        proj = FaultSubspaceProjector().fit(m_calc, dp_man)
        assert proj.rank == 3

        props = proj.verify_properties()
        assert props["is_valid"] is True
        assert props["idempotence_error"] < 1e-12
        assert props["symmetry_error"] < 1e-12
        assert props["orthogonality_error"] < 1e-12

        # Any linear combination of fault modes must be annihilated by P_{F^perp}
        r_fault = 0.002 - 0.04 * m_calc - 0.0015 * np.sqrt(dp_man / 50000.0)
        r_fault_perp = proj.project_residuals(r_fault)
        assert np.linalg.norm(r_fault_perp) < 1e-12

    def test_anti_masking_surrogate_zero_overlap(self) -> None:
        rng = np.random.default_rng(1337)
        n = 300
        rpm = rng.uniform(700.0, 3600.0, n)
        lvvt_m = rng.uniform(0.0003, 0.0090, n)
        m_calc = (rpm / 6000.0) * (lvvt_m / 0.0097) * 0.045 + 0.003
        dp_man = rng.uniform(4500.0, 48000.0, n)

        proj = FaultSubspaceProjector().fit(m_calc, dp_man)
        r_clean = proj.project_residuals(0.0008 * np.sin(rpm / 120.0) * np.cos(lvvt_m / 0.002))
        r_fault = 0.0010 - 0.05 * m_calc - 0.0012 * np.sqrt(dp_man / 50000.0)
        r_raw = r_clean + r_fault

        X = np.column_stack([rpm, lvvt_m])
        surrogate = OrthogonalPolySurrogate(degree=3, ridge_alpha=1e-3).fit(X, r_raw, m_calc, dp_man)
        g_perp = surrogate.predict_surrogate(X, m_calc, dp_man)
        F = FaultSubspaceProjector.build_fault_basis(m_calc, dp_man)
        audit = surrogate.audit_fault_non_absorption(F, g_perp)

        assert audit.is_anti_masking_verified is True
        assert audit.fault_subspace_norm < 1e-10

    def test_sample_count_validation(self) -> None:
        with pytest.raises(ValueError, match="must be >= rank"):
            FaultSubspaceProjector().fit([0.01, 0.02], [5000.0, 6000.0])
