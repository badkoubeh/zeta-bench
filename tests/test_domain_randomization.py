"""Unit tests for the training-time domain-randomisation wrapper.

These use a lightweight recording env stub (no dynamics / torch) so the wrapper's
config handling, sampling bounds, and seeding can be tested in isolation.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from omegaconf import OmegaConf

from envs.domain_randomization import DomainRandomizationWrapper, wrap_if_enabled


class _RecordingEnv(gym.Env):
    """Minimal env stub that records the most recent ``set_disturbance`` call."""

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.last_disturbance: dict | None = None

    def set_disturbance(self, **kwargs) -> None:
        self.last_disturbance = kwargs

    def reset(self, *, seed=None, options=None):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


def _dr_cfg(**overrides):
    """A full, valid ``env.domain_randomization`` block, with optional overrides."""
    block = {
        "enabled": True,
        "severity_anneal_steps": 0,
        "wind_magnitude_mps": [0.0, 10.0],
        "mass_offset_fraction": [-0.20, 0.20],
        "sensor_noise_sigma": [0.0, 0.03],
        "sensor_spike_probability": [0.0, 0.03],
        "sensor_spike_magnitude": 0.5,
        "actuator_delay_steps": [0, 0],
    }
    block.update(overrides)
    return OmegaConf.create(block)


def test_wrap_if_enabled_is_noop_when_disabled() -> None:
    """A disabled block leaves the env untouched (identity)."""
    env = _RecordingEnv()
    cfg = OmegaConf.create({"env": {"domain_randomization": _dr_cfg(enabled=False)}})
    assert wrap_if_enabled(env, cfg) is env


def test_wrap_if_enabled_is_noop_when_block_absent() -> None:
    """No domain_randomization block → env returned unchanged."""
    env = _RecordingEnv()
    cfg = OmegaConf.create({"env": {}})
    assert wrap_if_enabled(env, cfg) is env


def test_wrap_if_enabled_wraps_when_enabled() -> None:
    """An enabled block wraps the env in the randomisation wrapper."""
    env = _RecordingEnv()
    cfg = OmegaConf.create({"env": {"domain_randomization": _dr_cfg()}})
    assert isinstance(wrap_if_enabled(env, cfg), DomainRandomizationWrapper)


@pytest.mark.parametrize(
    "missing", ["wind_magnitude_mps", "sensor_noise_sigma", "sensor_spike_magnitude"]
)
def test_missing_range_raises_rather_than_defaulting(missing: str) -> None:
    """A missing magnitude is a config error, not a silent hardcoded fallback.

    Regression guard: config (configs/env.yaml) is the single source of disturbance
    magnitudes, so an absent key must fail loud instead of quietly training the
    policy on an unintended disturbance.
    """
    block = _dr_cfg()
    del block[missing]
    with pytest.raises(ValueError, match=missing):
        DomainRandomizationWrapper(_RecordingEnv(), block)


def test_zero_range_leaves_channel_nominal() -> None:
    """Setting a channel to [0.0, 0.0] disables it explicitly (nominal)."""
    env = _RecordingEnv()
    cfg = _dr_cfg(
        wind_magnitude_mps=[0.0, 0.0],
        mass_offset_fraction=[0.0, 0.0],
        sensor_noise_sigma=[0.0, 0.0],
        sensor_spike_probability=[0.0, 0.0],
    )
    DomainRandomizationWrapper(env, cfg).reset(seed=0)
    d = env.last_disturbance
    assert d["wind_velocity_ned"] is None
    assert d["mass_offset_fraction"] == 0.0
    assert d["sensor_noise_sigma"] == 0.0
    assert d["sensor_spike_probability"] == 0.0


def test_sampled_disturbance_stays_within_config_ranges() -> None:
    """Every sampled disturbance respects the configured bounds (no divergence)."""
    env = _RecordingEnv()
    wrapper = DomainRandomizationWrapper(env, _dr_cfg())
    for seed in range(200):
        wrapper.reset(seed=seed)
        d = env.last_disturbance
        assert 0.0 <= d["sensor_noise_sigma"] <= 0.03
        assert 0.0 <= d["sensor_spike_probability"] <= 0.03
        assert -0.20 <= d["mass_offset_fraction"] <= 0.20
        wind = d["wind_velocity_ned"]
        if wind is not None:
            assert float(np.linalg.norm(wind)) <= 10.0 + 1e-9


def test_seeded_reset_is_deterministic() -> None:
    """Two wrappers reset with the same seed sample the same disturbance."""
    env_a, env_b = _RecordingEnv(), _RecordingEnv()
    DomainRandomizationWrapper(env_a, _dr_cfg()).reset(seed=123)
    DomainRandomizationWrapper(env_b, _dr_cfg()).reset(seed=123)
    a, b = env_a.last_disturbance, env_b.last_disturbance
    assert a["sensor_noise_sigma"] == b["sensor_noise_sigma"]
    assert a["sensor_spike_probability"] == b["sensor_spike_probability"]
    assert a["mass_offset_fraction"] == b["mass_offset_fraction"]


def test_severity_anneal_scales_ranges_upward() -> None:
    """Early in the severity ramp, sampled magnitudes are compressed toward nominal."""
    env = _RecordingEnv()
    wrapper = DomainRandomizationWrapper(env, _dr_cfg(severity_anneal_steps=1000))
    # At step 0 severity is 0 → sigma upper bound collapses to 0.
    wrapper.reset(seed=1)
    assert env.last_disturbance["sensor_noise_sigma"] == 0.0
