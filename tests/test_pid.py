"""Unit tests for :class:`controllers.pid_baseline.PIDController`."""
from __future__ import annotations

import numpy as np
import pytest
from hydra import compose, initialize

from controllers.pid_baseline import PIDController
from envs.rocket_landing_env import OBS_DIM, RocketLandingEnv


@pytest.fixture
def cfg():
    with initialize(config_path="../configs", version_base=None):
        return compose(config_name="train")


def test_pid_predict_returns_3dim_action(cfg) -> None:
    """predict() returns a 3-vector regardless of input observation."""
    pid = PIDController(cfg)
    obs = np.zeros(OBS_DIM)
    action = pid.predict(obs)
    assert action.shape == (3,)
    assert action.dtype == np.float64


def test_pid_action_in_env_action_space(cfg) -> None:
    """Whatever PID outputs, it must lie inside the env's action_space box."""
    env = RocketLandingEnv(cfg)
    pid = PIDController(cfg)
    obs, _ = env.reset(seed=42)
    pid.reset()
    action = pid.predict(obs)
    # Throttle ∈ [0, 1]
    assert 0.0 <= action[0] <= 1.0
    # Gimbal commands ∈ [-1, 1]
    assert -1.0 <= action[1] <= 1.0
    assert -1.0 <= action[2] <= 1.0


def test_pid_reset_clears_integrator_state(cfg) -> None:
    """After reset(), the integrator and previous-error state are zero."""
    pid = PIDController(cfg)
    # Drive the integrator by running predict() a few times with a velocity error
    obs = np.zeros(OBS_DIM)
    obs[5] = 50.0  # large vz error (in scaled units; the unscale multiplies)
    for _ in range(10):
        pid.predict(obs)
    assert pid._alt_integral != 0.0  # noqa: SLF001 (internal state check)

    pid.reset()
    assert pid._alt_integral == 0.0  # noqa: SLF001
    assert pid._prev_vz_error == 0.0  # noqa: SLF001


def test_pid_runs_full_episode_without_exception(cfg) -> None:
    """End-to-end smoke: env.reset → PID.predict → env.step loop for
    `max_steps` iterations completes without an exception.

    This does NOT assert the rocket actually lands — that requires tuned
    gains — a follow-up task. We only assert the pipeline
    integrates cleanly.
    """
    env = RocketLandingEnv(cfg)
    pid = PIDController(cfg)
    obs, _ = env.reset(seed=42)
    pid.reset()

    for _ in range(cfg.env.episode.max_steps):
        action = pid.predict(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    # If we get here, no exception was raised — the loop is integrating cleanly.
