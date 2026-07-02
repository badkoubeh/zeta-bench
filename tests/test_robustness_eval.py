"""Tests for the graduated disturbance-matrix runner and heatmap.

Exercises :func:`robustness.evaluation.run_matrix` on a shrunk grid with a real
PID controller plus a stateless stub controller (covering the with/without
``reset()`` paths), the CSV writer, the empty-input summary guard, the
fixed-vs-linear schedule fairness warning, and the heatmap renderer.

Uses a real (physics-backed) env with ``max_steps`` shrunk so the matrix runs in
a second or two — this is a wiring smoke test, not a controller benchmark.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from controllers.pid_baseline import PIDController
from robustness.evaluation import (
    MATRIX_COLUMNS,
    run_matrix,
    summarise,
    write_matrix_csv,
)
from robustness.heatmap import plot_robustness_heatmap


@pytest.fixture(autouse=True)
def _clear_hydra():
    """Reset Hydra's global singleton around each test (compose hygiene)."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    yield
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


_SMALL_GRID = [
    "env.curriculum.schedule=fixed",
    "env.curriculum.task_difficulty=0.0",
    "env.episode.max_steps=40",
    "eval.seeds=1",
    "eval.episodes_per_seed=2",
    "eval.disturbance_grid.wind.magnitudes_mps=[0.0,5.0]",
    "eval.disturbance_grid.wind.directions_deg=[0,90]",
    "eval.disturbance_grid.mass_offset_fraction=[0.0,0.1]",
    "eval.disturbance_grid.sensor_noise.sigma=[0.0,0.05]",
    "eval.disturbance_grid.sensor_noise.spike_probability=[0.0]",
]


def _cfg(overrides):
    with initialize(config_path="../configs", version_base=None):
        return compose(config_name="train", overrides=overrides)


class _ConstantController:
    """Stateless stub controller (no ``reset``) — exercises the uniform path."""

    def predict(self, obs, deterministic: bool = True) -> np.ndarray:
        return np.array([0.5, 0.0, 0.0], dtype=np.float64)


# --- summarise guard -------------------------------------------------------

def test_summarise_empty_returns_zeros() -> None:
    """Aggregating zero episodes yields an all-zero summary (no div-by-zero)."""
    s = summarise([])
    assert s["n_episodes"] == 0
    assert s["success_rate"] == 0.0
    assert s["touchdown_speed_mean_mps"] == 0.0


def test_summarise_counts_outcomes() -> None:
    """Success rate and outcome counts aggregate correctly."""
    rows = [
        {"outcome": "success", "return": 1.0, "length": 10,
         "touchdown_speed_mps": 1.0, "fuel_used_kg": 5.0},
        {"outcome": "crash", "return": -1.0, "length": 8,
         "touchdown_speed_mps": 9.0, "fuel_used_kg": 6.0},
    ]
    s = summarise(rows)
    assert s["n_episodes"] == 2
    assert s["success_rate"] == 0.5
    assert s["n_success"] == 1 and s["n_crash"] == 1


# --- run_matrix ------------------------------------------------------------

def test_run_matrix_shape_and_fairness(tmp_path) -> None:
    """Matrix produces one row per (controller, cell) with valid success rates."""
    cfg = _cfg(_SMALL_GRID)
    controllers = {"pid": PIDController(cfg), "const": _ConstantController()}
    rows = run_matrix(cfg, controllers)

    # 6 cells (nominal + 2 wind + 1 mass + 1 sensor_noise + 1 combined) × 2 controllers.
    n_cells = len({r["label"] for r in rows})
    assert n_cells == 6
    assert len(rows) == n_cells * 2

    for r in rows:
        assert set(MATRIX_COLUMNS).issubset(r.keys())
        assert 0.0 <= float(r["success_rate"]) <= 1.0
        assert int(r["n_episodes"]) == 2
    assert {"nominal", "wind", "mass", "sensor_noise", "combined"} == {
        r["disturbance_type"] for r in rows
    }


def test_run_matrix_identical_seeds_across_controllers(tmp_path) -> None:
    """Two identical controllers see identical per-cell results (fixed seeds)."""
    cfg = _cfg(_SMALL_GRID)
    controllers = {"a": _ConstantController(), "b": _ConstantController()}
    rows = run_matrix(cfg, controllers)
    by_label_a = {r["label"]: r for r in rows if r["controller"] == "a"}
    by_label_b = {r["label"]: r for r in rows if r["controller"] == "b"}
    for label, ra in by_label_a.items():
        rb = by_label_b[label]
        # Identical controllers under identical seeds => identical outcomes.
        assert ra["success_rate"] == rb["success_rate"]
        assert ra["return_mean"] == rb["return_mean"]


def test_run_matrix_warns_on_non_fixed_schedule(caplog) -> None:
    """A non-fixed curriculum schedule triggers the fairness warning."""
    cfg = _cfg([
        "env.curriculum.schedule=linear",
        "env.episode.max_steps=20",
        "eval.seeds=1",
        "eval.episodes_per_seed=1",
        "eval.disturbance_grid.wind.magnitudes_mps=[0.0]",
        "eval.disturbance_grid.mass_offset_fraction=[0.0]",
        "eval.disturbance_grid.sensor_noise.sigma=[0.0]",
        "eval.disturbance_grid.sensor_noise.spike_probability=[0.0]",
        "eval.disturbance_grid.combined.enabled=false",
    ])
    with caplog.at_level("WARNING"):
        rows = run_matrix(cfg, {"const": _ConstantController()})
    assert any("schedule" in rec.message for rec in caplog.records)
    assert len(rows) == 1  # nominal-only grid


# --- CSV output ------------------------------------------------------------

def test_write_matrix_csv_roundtrip(tmp_path) -> None:
    """CSV is written with the canonical header and one row per matrix row."""
    cfg = _cfg(_SMALL_GRID)
    rows = run_matrix(cfg, {"const": _ConstantController()})
    out = write_matrix_csv(rows, tmp_path / "robustness_matrix.csv")
    assert out.exists()

    with out.open() as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames) == MATRIX_COLUMNS
        parsed = list(reader)
    assert len(parsed) == len(rows)
    # None-valued axis columns serialise to empty strings.
    nominal = next(p for p in parsed if p["disturbance_type"] == "nominal")
    assert nominal["wind_direction_deg"] == ""


# --- heatmap ---------------------------------------------------------------

def test_plot_heatmap_writes_png(tmp_path) -> None:
    """The heatmap renders a non-empty PNG for a real matrix result."""
    cfg = _cfg(_SMALL_GRID)
    rows = run_matrix(cfg, {"pid": PIDController(cfg), "const": _ConstantController()})
    out = plot_robustness_heatmap(rows, tmp_path / "heatmap.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_heatmap_infers_episode_count_and_handles_missing_nominal(tmp_path) -> None:
    """Heatmap infers n from rows and tolerates rows without a nominal cell."""
    rows = [
        {"controller": "x", "disturbance_type": "wind", "severity": 5.0,
         "success_rate": 0.8, "n_episodes": 3},
        {"controller": "x", "disturbance_type": "wind", "severity": 10.0,
         "success_rate": 0.4, "n_episodes": 3},
        {"controller": "x", "disturbance_type": "mass", "severity": 0.1,
         "success_rate": 0.6, "n_episodes": 3},
    ]
    out = plot_robustness_heatmap(rows, tmp_path / "synthetic.png")
    assert out.exists()


# --- entrypoint smoke ------------------------------------------------------

def test_evaluate_robustness_entrypoint(tmp_path) -> None:
    """The Hydra entrypoint writes a matrix CSV + heatmap for a PID-only sweep."""
    from experiments.evaluate_robustness import main as evaluate_robustness_main

    csv_path = tmp_path / "robustness_matrix.csv"
    png_path = tmp_path / "robustness_heatmap.png"
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(
            config_name="eval_robustness",
            overrides=[
                "eval_robustness.controllers.sac.enabled=false",
                "eval_robustness.controllers.ppo.enabled=false",
                "env.episode.max_steps=40",
                "eval.seeds=1",
                "eval.episodes_per_seed=2",
                "eval.disturbance_grid.wind.magnitudes_mps=[0.0,5.0]",
                "eval.disturbance_grid.wind.directions_deg=[0]",
                "eval.disturbance_grid.mass_offset_fraction=[0.0,0.1]",
                "eval.disturbance_grid.sensor_noise.sigma=[0.0]",
                "eval.disturbance_grid.sensor_noise.spike_probability=[0.0]",
                "eval.disturbance_grid.combined.enabled=false",
                "eval.outputs.log_wandb_table=false",
                f"eval.outputs.csv_path={csv_path}",
                f"eval.outputs.heatmap_path={png_path}",
            ],
        )
    evaluate_robustness_main.__wrapped__(cfg)  # bypass Hydra's CLI shim

    assert csv_path.exists()
    assert png_path.exists()
    with csv_path.open() as f:
        parsed = list(csv.DictReader(f))
    # 3 cells (nominal + 1 wind + 1 mass) × PID only.
    assert {p["disturbance_type"] for p in parsed} == {"nominal", "wind", "mass"}
    assert all(p["controller"] == "pid" for p in parsed)
