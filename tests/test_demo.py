"""Smoke tests for the runnable showcase demo script."""

from __future__ import annotations

from scripts.run_showcase_demo import main, run_showcase


def test_run_showcase_metrics() -> None:
    payload = run_showcase()
    assert payload["iso19848_polymorphic_twin"]["automotive"]["plant_id"] == "BMW_N42B20A"
    assert payload["iso19848_polymorphic_twin"]["maritime"]["plant_id"] == "MAN_BW_6S50ME_C"
    assert payload["multi_regime_observability"]["single_regime_rank"] == 4
    assert payload["multi_regime_observability"]["full_stack_rank"] == 8
    assert payload["multi_regime_observability"]["jacobian_gate_passed"] is True
    assert payload["orthogonal_projector"]["anti_masking_verified"] is True
    assert payload["orthogonal_projector"]["orthogonality_error"] < 1e-12


def test_demo_cli_exit_code() -> None:
    assert main(["--json"]) == 0
