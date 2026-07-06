"""Tests for disturbance injection at the :class:`RocketLandingEnv` boundary.

Covers the primitive ``set_disturbance`` hook and each disturbance's observable
effect: sensor noise (reproducible under a fixed seed; a no-op at σ=0), actuator
delay (applied action lags the command), mass offset (rebuilds the dynamics),
and wind (alters the trajectory). Also checks the nominal env is byte-for-byte
unchanged, guarding the "no disturbance leaks into nominal runs" invariant.
"""
from __future__ import annotations

import numpy as np
import pytest
from hydra import compose, initialize

from envs.rocket_landing_env import OBS_DIM, RocketLandingEnv


@pytest.fixture
def cfg():
    with initialize(config_path="../configs", version_base=None):
        return compose(config_name="train")


def _rollout_obs(env: RocketLandingEnv, seed: int, steps: int, action) -> list[np.ndarray]:
    obs, _ = env.reset(seed=seed)
    frames = [obs]
    for _ in range(steps):
        obs, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float64))
        frames.append(obs)
        if terminated or truncated:
            break
    return frames


# --- nominal invariance ----------------------------------------------------

def test_default_env_is_nominal(cfg) -> None:
    """With no disturbance configured, all disturbance state is nominal."""
    env = RocketLandingEnv(cfg)
    assert env._wind_velocity_ned is None
    assert env._mass_offset_fraction == 0.0
    assert env._sensor_noise_sigma == 0.0
    assert env._actuator_delay_steps == 0


# --- sensor noise ----------------------------------------------------------

def test_sensor_noise_zero_sigma_is_noop(cfg) -> None:
    """σ=0 and spike_prob=0 leaves the observation identical to nominal."""
    nominal = RocketLandingEnv(cfg)
    noisy = RocketLandingEnv(cfg)
    noisy.set_disturbance(sensor_noise_sigma=0.0, sensor_spike_probability=0.0)
    a, _ = nominal.reset(seed=7)
    b, _ = noisy.reset(seed=7)
    np.testing.assert_array_equal(a, b)


def test_sensor_noise_perturbs_observation(cfg) -> None:
    """A non-zero σ changes the observation away from the nominal one."""
    nominal = RocketLandingEnv(cfg)
    noisy = RocketLandingEnv(cfg)
    noisy.set_disturbance(sensor_noise_sigma=0.1)
    a, _ = nominal.reset(seed=7)
    b, _ = noisy.reset(seed=7)
    assert a.shape == b.shape == (OBS_DIM,)
    assert not np.allclose(a, b)


def test_sensor_noise_is_reproducible_under_seed(cfg) -> None:
    """Same seed + same σ reproduces the exact noisy observation sequence."""
    env_a = RocketLandingEnv(cfg)
    env_b = RocketLandingEnv(cfg)
    env_a.set_disturbance(sensor_noise_sigma=0.05, sensor_spike_probability=0.02,
                          sensor_spike_magnitude=0.5)
    env_b.set_disturbance(sensor_noise_sigma=0.05, sensor_spike_probability=0.02,
                          sensor_spike_magnitude=0.5)
    frames_a = _rollout_obs(env_a, seed=11, steps=5, action=[0.5, 0.0, 0.0])
    frames_b = _rollout_obs(env_b, seed=11, steps=5, action=[0.5, 0.0, 0.0])
    assert len(frames_a) == len(frames_b)
    for fa, fb in zip(frames_a, frames_b):
        np.testing.assert_array_equal(fa, fb)


# --- actuator delay --------------------------------------------------------

def test_actuator_delay_lags_applied_action(cfg) -> None:
    """With delay=2 the obs last_action (indices 12:15) lags the command by 2 ticks.

    The neutral cold-start means the first two applied actions are zero, then
    the commands begin to surface.
    """
    env = RocketLandingEnv(cfg)
    env.set_disturbance(actuator_delay_steps=2)
    env.reset(seed=3)
    cmd = np.array([0.8, 0.5, -0.5])
    # Step 1 & 2: applied action is still the neutral cold-start (zeros).
    obs1, *_ = env.step(cmd)
    obs2, *_ = env.step(cmd)
    # last_action is stored in the scaled obs at indices 12:15 (passthrough scale).
    np.testing.assert_allclose(obs1[12:15], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(obs2[12:15], [0.0, 0.0, 0.0], atol=1e-6)
    # Step 3: the first command finally reaches the actuator.
    obs3, *_ = env.step(cmd)
    assert not np.allclose(obs3[12:15], [0.0, 0.0, 0.0], atol=1e-6)


def test_actuator_delay_zero_applies_command_immediately(cfg) -> None:
    """delay=0 applies the command on the same tick (no buffer)."""
    env = RocketLandingEnv(cfg)
    env.set_disturbance(actuator_delay_steps=0)
    env.reset(seed=3)
    cmd = np.array([0.8, 0.5, -0.5])
    obs, *_ = env.step(cmd)
    assert not np.allclose(obs[12:15], [0.0, 0.0, 0.0], atol=1e-6)


# --- mass offset -----------------------------------------------------------

def test_mass_offset_rebuilds_dynamics(cfg) -> None:
    """A mass offset scales the dynamics dry mass by (1 + fraction)."""
    env = RocketLandingEnv(cfg)
    base_dry = env._base_params.dry_mass_kg
    env.set_disturbance(mass_offset_fraction=0.20)
    assert env._mass_offset_fraction == 0.20
    assert np.isclose(env._dynamics.get_params()[0], base_dry * 1.20)


def test_mass_offset_changes_trajectory(cfg) -> None:
    """A heavier vehicle follows a different trajectory under identical control."""
    light = RocketLandingEnv(cfg)
    heavy = RocketLandingEnv(cfg)
    heavy.set_disturbance(mass_offset_fraction=0.20)
    a = _rollout_obs(light, seed=5, steps=10, action=[0.6, 0.0, 0.0])
    b = _rollout_obs(heavy, seed=5, steps=10, action=[0.6, 0.0, 0.0])
    assert not np.allclose(a[-1], b[-1])


# --- wind ------------------------------------------------------------------

def test_wind_changes_trajectory(cfg) -> None:
    """A crosswind alters the trajectory relative to the nominal env."""
    nominal = RocketLandingEnv(cfg)
    windy = RocketLandingEnv(cfg)
    windy.set_disturbance(wind_velocity_ned=np.array([0.0, 8.0, 0.0]))
    a = _rollout_obs(nominal, seed=5, steps=10, action=[0.6, 0.0, 0.0])
    b = _rollout_obs(windy, seed=5, steps=10, action=[0.6, 0.0, 0.0])
    assert not np.allclose(a[-1], b[-1])


# --- config-driven construction -------------------------------------------

def test_disturbance_from_config_block(cfg) -> None:
    """An env.disturbance config block is applied at construction."""
    cfg.env.disturbance.wind_magnitude_mps = 5.0
    cfg.env.disturbance.wind_direction_deg = 90.0
    cfg.env.disturbance.mass_offset_fraction = 0.1
    cfg.env.disturbance.actuator_delay_steps = 3
    env = RocketLandingEnv(cfg)
    np.testing.assert_allclose(env._wind_velocity_ned, [0.0, 5.0, 0.0], atol=1e-9)
    assert np.isclose(env._mass_offset_fraction, 0.1)
    assert env._actuator_delay_steps == 3
