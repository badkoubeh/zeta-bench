"""Unit tests for :class:`envs.rocket_landing_env.RocketLandingEnv`."""
from __future__ import annotations

import numpy as np
import pytest
from hydra import compose, initialize

from dynamics.types import UPRIGHT_QUAT, make_state
from envs.rocket_landing_env import OBS_DIM, RocketLandingEnv


@pytest.fixture
def cfg():
    with initialize(config_path="../configs", version_base=None):
        return compose(config_name="train")


def test_env_constructs_without_error(cfg) -> None:
    """Constructor succeeds with the default config tree."""
    env = RocketLandingEnv(cfg)
    assert env.observation_space.shape == (OBS_DIM,)
    assert env.action_space.shape == (3,)


def test_reset_returns_valid_obs_and_info(cfg) -> None:
    """reset() returns a 17-dim obs and an info dict carrying curriculum progress."""
    env = RocketLandingEnv(cfg)
    obs, info = env.reset(seed=42)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert "curriculum_progress" in info
    assert 0.0 <= info["curriculum_progress"] <= 1.0


def test_reset_with_seed_is_deterministic(cfg) -> None:
    """Two resets with the same seed yield identical observations."""
    env_a = RocketLandingEnv(cfg)
    env_b = RocketLandingEnv(cfg)
    obs_a, _ = env_a.reset(seed=42)
    obs_b, _ = env_b.reset(seed=42)
    assert np.allclose(obs_a, obs_b)


def test_step_returns_valid_five_tuple(cfg) -> None:
    """step() returns (obs, reward, terminated, truncated, info) with correct shapes/types."""
    env = RocketLandingEnv(cfg)
    env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(np.array([0.5, 0.0, 0.0]))
    assert obs.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert "reward_components" in info


def test_action_clipped_to_bounds(cfg) -> None:
    """Out-of-bounds actions are clipped silently — the env doesn't raise."""
    env = RocketLandingEnv(cfg)
    env.reset(seed=42)
    # Action with throttle > 1 and gimbal < -1 should be clipped
    obs, *_ = env.step(np.array([5.0, -3.0, 2.0]))
    assert obs.shape == (OBS_DIM,)


def test_terminal_on_crash_when_dropped_fast_onto_pad(cfg) -> None:
    """Hand-engineer a touchdown with high vertical velocity — env returns
    terminated=True with reason 'crash'.
    """
    env = RocketLandingEnv(cfg)
    env.reset(seed=42)
    # Directly set the internal state to be just above the pad with high vz
    env._state = make_state(
        position_NED=np.array([0.0, 0.0, -0.1]),  # ~10 cm above pad
        velocity_NED=np.array([0.0, 0.0, 50.0]),  # 50 m/s downward — way above 2 m/s threshold
        quat_wxyz=UPRIGHT_QUAT,
        angular_rate_body=np.zeros(3),
        fuel_mass_kg=4000.0,
    )
    _, _, terminated, truncated, info = env.step(np.array([0.0, 0.0, 0.0]))
    assert terminated is True
    assert truncated is False
    assert info["termination_reason"] == "crash"


def test_terminal_on_out_of_bounds_lateral(cfg) -> None:
    """Position outside the cylinder radius ⇒ terminated with reason 'out_of_bounds'."""
    env = RocketLandingEnv(cfg)
    env.reset(seed=42)
    env._state = make_state(
        position_NED=np.array([500.0, 0.0, -100.0]),  # 500 m east, well beyond 200 m radius
        velocity_NED=np.zeros(3),
        quat_wxyz=UPRIGHT_QUAT,
        angular_rate_body=np.zeros(3),
        fuel_mass_kg=4000.0,
    )
    _, _, terminated, _, info = env.step(np.zeros(3))
    assert terminated is True
    assert info["termination_reason"] == "out_of_bounds"


def test_terminal_on_successful_soft_landing(cfg) -> None:
    """Engineered: rocket touches the pad (z = 0) with velocity, tilt, ω all
    below their thresholds. Env returns terminated with reason 'success'.
    """
    env = RocketLandingEnv(cfg)
    env.reset(seed=42)
    # Start exactly at the pad surface. The single-step descent at 1 m/s
    # over 20 ms (~2 cm) takes z just past zero, which triggers the
    # touchdown check; vz / tilt / ω are all under their success thresholds.
    env._state = make_state(
        position_NED=np.array([0.0, 0.0, 0.0]),
        velocity_NED=np.array([0.0, 0.0, 1.0]),  # 1 m/s, under 2 m/s threshold
        quat_wxyz=UPRIGHT_QUAT,
        angular_rate_body=np.zeros(3),
        fuel_mass_kg=4000.0,
    )
    _, reward, terminated, _, info = env.step(np.array([0.5, 0.0, 0.0]))
    assert terminated is True
    assert info["termination_reason"] == "success"
    # Reward should be positive (dominated by the success bonus)
    assert reward > 0


def test_step_before_reset_raises(cfg) -> None:
    """Calling step() before reset() is a programmer error and must raise."""
    env = RocketLandingEnv(cfg)
    with pytest.raises(RuntimeError):
        env.step(np.zeros(3))
