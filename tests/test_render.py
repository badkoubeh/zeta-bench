"""Smoke tests for :mod:`utils.render`.

Build a tiny synthetic trajectory and verify both visualization entry
points produce non-empty output files.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from utils.render import Trajectory, animate_side_view, plot_timeseries


def _make_synthetic_trajectory(T: int = 10) -> Trajectory:
    """A descending-but-tilted rocket: useful for eyeballing rotation logic."""
    t = np.linspace(0.0, (T - 1) * 0.1, T)
    altitude = np.linspace(200.0, 50.0, T)
    pos_NED = np.zeros((T, 3), dtype=np.float64)
    pos_NED[:, 0] = np.linspace(0.0, 20.0, T)  # drifts right
    pos_NED[:, 2] = -altitude
    vel_NED = np.zeros((T, 3), dtype=np.float64)
    vel_NED[:, 0] = 2.0
    vel_NED[:, 2] = 15.0
    euler = np.zeros((T, 3), dtype=np.float64)
    euler[:, 1] = np.linspace(0.0, 0.2, T)  # pitch increases
    omega_body = np.zeros((T, 3), dtype=np.float64)
    action = np.zeros((T, 3), dtype=np.float64)
    action[:, 0] = np.linspace(0.4, 1.0, T)  # throttle ramp
    reward = -np.ones(T, dtype=np.float64)
    fuel_kg = np.linspace(5000.0, 4500.0, T)

    return Trajectory(
        t=t,
        pos_NED=pos_NED,
        vel_NED=vel_NED,
        euler=euler,
        omega_body=omega_body,
        action=action,
        reward=reward,
        fuel_kg=fuel_kg,
        meta={
            "outcome": "synthetic",
            "episode_idx": 0,
            "seed": 0,
            "return_total": float(reward.sum()),
            "pad_radius_m": 30.0,
            "oob_cylinder_radius_m": 200.0,
            "oob_ceiling_m": 600.0,
            "target_descent_mps": 2.0,
        },
    )


def test_plot_timeseries_writes_png(tmp_path: Path) -> None:
    traj = _make_synthetic_trajectory(T=10)
    out = tmp_path / "timeseries.png"
    plot_timeseries(traj, out)
    assert out.exists()
    assert out.stat().st_size > 1024  # not empty/blank


@pytest.mark.skipif(
    importlib.util.find_spec("imageio_ffmpeg") is None,
    reason="imageio_ffmpeg not installed",
)
def test_animate_side_view_writes_mp4(tmp_path: Path) -> None:
    traj = _make_synthetic_trajectory(T=10)
    out = tmp_path / "landing.mp4"
    animate_side_view(traj, out, fps=5)
    assert out.exists()
    assert out.stat().st_size > 1024
