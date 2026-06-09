"""Smoke test for :mod:`experiments.evaluate_pid`.

Composes the eval_pid config, calls the entrypoint with a tiny episode
budget, and asserts the two output files exist and parse.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from experiments.evaluate_pid import main as evaluate_pid_main


@pytest.fixture(autouse=True)
def _clear_hydra():
    """Hydra's global state leaks across tests; clear before and after."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    yield
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


def test_evaluate_pid_writes_expected_outputs(tmp_path: Path) -> None:
    """End-to-end: script runs with minimal budget; CSV + JSON are produced."""
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(
            config_name="eval_pid",
            overrides=[
                "eval_pid.n_episodes=2",
                "env.episode.max_steps=50",
                f"results_dir={tmp_path}",
            ],
        )

    evaluate_pid_main.__wrapped__(cfg)  # bypass Hydra's CLI shim

    episodes_csv = tmp_path / "episodes.csv"
    summary_json = tmp_path / "summary.json"

    assert episodes_csv.exists(), "episodes.csv was not written"
    assert summary_json.exists(), "summary.json was not written"

    with summary_json.open() as f:
        summary = json.load(f)

    assert summary["n_episodes"] == 2
    for key in (
        "success_rate",
        "return_mean",
        "return_std",
        "touchdown_speed_mean_mps",
        "fuel_used_mean_kg",
        "episode_length_mean",
    ):
        assert key in summary, f"summary missing key: {key}"


def test_evaluate_pid_render_writes_plots_and_video(tmp_path: Path) -> None:
    """With render=true, PNG + MP4 are written for best and worst episodes."""
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(
            config_name="eval_pid",
            overrides=[
                "eval_pid.n_episodes=2",
                "eval_pid.render=true",
                "eval_pid.render_fps=10",
                "env.episode.max_steps=20",
                f"results_dir={tmp_path}",
            ],
        )

    evaluate_pid_main.__wrapped__(cfg)

    plots = list((tmp_path / "plots").glob("timeseries_*.png"))
    videos = list((tmp_path / "video").glob("landing_*.mp4"))

    assert len(plots) >= 1, "no timeseries PNGs were written"
    assert len(videos) >= 1, "no landing MP4s were written"
    for p in plots + videos:
        assert p.stat().st_size > 1024, f"{p} is suspiciously small"
