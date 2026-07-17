"""Tests for the per-controller robustness cards.

Exercises curve aggregation (secondary-axis averaging, nominal anchoring,
signed severities), break-point semantics (first failing magnitude,
right-censoring), summary assembly with missing variants, and the PNG/JSON
writers on synthetic rows shaped exactly like ``csv.DictReader`` output
(string values), plus an integration pass over the real matrix CSV when it
exists in ``results/``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from robustness.cards import (
    break_point,
    build_card_summary,
    degradation_curve,
    load_matrix_rows,
    plot_robustness_card,
    write_card_summary,
)

_REAL_CSV = Path(__file__).resolve().parents[1] / "results" / "robustness_matrix.csv"


def _row(
    controller: str,
    dtype: str,
    severity: float,
    success: float,
    **extra: object,
) -> dict[str, object]:
    """One matrix row as csv.DictReader would yield it (all-string values)."""
    row: dict[str, object] = {
        "controller": controller,
        "disturbance_type": dtype,
        "severity": str(severity),
        "success_rate": str(success),
        "n_episodes": "100",
    }
    row.update({k: str(v) for k, v in extra.items()})
    return row


def _synthetic_rows() -> list[dict[str, object]]:
    return [
        _row("sac", "nominal", 0.0, 1.0),
        # wind: two magnitudes x two directions -> mean over direction
        _row("sac", "wind", 5.0, 1.0, wind_direction_deg=0.0),
        _row("sac", "wind", 5.0, 0.9, wind_direction_deg=90.0),
        _row("sac", "wind", 10.0, 0.5, wind_direction_deg=0.0),
        _row("sac", "wind", 10.0, 0.3, wind_direction_deg=90.0),
        # mass: signed severities, negative side fails first
        _row("sac", "mass", -0.2, 0.2),
        _row("sac", "mass", -0.1, 0.9),
        _row("sac", "mass", 0.1, 1.0),
        _row("sac", "mass", 0.2, 1.0),
        # sensor noise: sigma-0 level exists (spike-only cell), so nominal is
        # NOT appended as an extra 0 anchor
        _row("sac", "sensor_noise", 0.0, 0.97, spike_probability=0.01),
        _row("sac", "sensor_noise", 0.05, 0.4, spike_probability=0.0),
        _row("sac", "combined", 10.0, 0.0),
    ]


class TestDegradationCurve:
    def test_wind_averages_directions_and_anchors_nominal(self):
        curve = degradation_curve(_synthetic_rows(), "sac", "wind")
        assert curve == [(0.0, 1.0), (5.0, pytest.approx(0.95)), (10.0, pytest.approx(0.4))]

    def test_mass_keeps_sign_and_inserts_nominal_at_zero(self):
        curve = degradation_curve(_synthetic_rows(), "sac", "mass")
        assert [sev for sev, _ in curve] == [-0.2, -0.1, 0.0, 0.1, 0.2]
        assert dict(curve)[0.0] == 1.0

    def test_sensor_noise_zero_level_not_overwritten_by_nominal(self):
        curve = degradation_curve(_synthetic_rows(), "sac", "sensor_noise")
        assert [sev for sev, _ in curve] == [0.0, 0.05]
        # sigma=0 point is the spike-only cell (0.97), not the nominal 1.0
        assert curve[0][1] == pytest.approx(0.97)

    def test_unknown_controller_or_family_is_empty(self):
        assert degradation_curve(_synthetic_rows(), "sac", "actuator_delay") == []
        assert degradation_curve(_synthetic_rows(), "lqr", "wind") == []


class TestBreakPoint:
    def test_first_failing_magnitude(self):
        curve = [(0.0, 1.0), (5.0, 0.95), (10.0, 0.4)]
        assert break_point(curve, gate=0.95) == (10.0, 10.0)

    def test_signed_family_breaks_on_failing_side(self):
        curve = degradation_curve(_synthetic_rows(), "sac", "mass")
        bp, max_tested = break_point(curve, gate=0.95)
        assert bp == 0.1  # -0.1 side fails (0.9 < 0.95) before +/-0.2
        assert max_tested == 0.2

    def test_censored_when_gate_holds_everywhere(self):
        curve = [(0.0, 1.0), (5.0, 1.0), (10.0, 0.99)]
        assert break_point(curve, gate=0.95) == (None, 10.0)

    def test_empty_curve(self):
        assert break_point([], gate=0.95) == (None, None)


class TestCardSummary:
    def test_missing_variant_skipped_present_variant_built(self):
        summary = build_card_summary(
            _synthetic_rows(),
            card="sac",
            variants={"nominal-trained": "sac", "robust": "sac_robust"},
            gate=0.95,
        )
        assert set(summary["variants"]) == {"nominal-trained"}
        variant = summary["variants"]["nominal-trained"]
        assert variant["nominal"] == 1.0
        assert variant["combined_max"] == 0.0
        assert set(variant["families"]) == {"wind", "mass", "sensor_noise"}
        assert variant["families"]["wind"]["break_point"] == 10.0

    def test_outputs_written_and_json_round_trips(self, tmp_path):
        summary = build_card_summary(
            _synthetic_rows(), card="sac", variants={"nominal-trained": "sac"}, gate=0.95
        )
        png = plot_robustness_card(summary, tmp_path / "sac.png", episodes_per_cell=100)
        jsn = write_card_summary(summary, tmp_path / "sac.json")
        assert png.exists() and png.stat().st_size > 0
        loaded = json.loads(jsn.read_text())
        assert loaded["gate"] == 0.95
        assert loaded["variants"]["nominal-trained"]["families"]["mass"]["break_point"] == 0.1

    def test_plot_rejects_empty_card(self, tmp_path):
        summary = build_card_summary(
            _synthetic_rows(), card="lqr", variants={"tuned": "lqr"}, gate=0.95
        )
        with pytest.raises(ValueError, match="no plottable"):
            plot_robustness_card(summary, tmp_path / "lqr.png")


class TestLoadMatrixRows:
    def test_missing_file_returns_empty(self, tmp_path):
        assert load_matrix_rows(tmp_path / "nope.csv") == []

    @pytest.mark.skipif(not _REAL_CSV.exists(), reason="results CSV not present")
    def test_real_matrix_builds_cards(self, tmp_path):
        rows = load_matrix_rows(_REAL_CSV)
        assert rows, "tracked matrix CSV should have rows"
        for controller in ("pid", "sac", "ppo"):
            summary = build_card_summary(
                rows, card=controller, variants={"nominal-trained": controller}, gate=0.95
            )
            assert summary["variants"], controller
            plot_robustness_card(summary, tmp_path / f"{controller}.png")
        # documented headline: SAC collapses under sensor noise -> break-point exists
        sac = build_card_summary(
            rows, card="sac", variants={"nominal-trained": "sac"}, gate=0.95
        )
        noise = sac["variants"]["nominal-trained"]["families"]["sensor_noise"]
        assert noise["break_point"] is not None
