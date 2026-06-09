"""Unit tests for :class:`envs.curriculum.Curriculum`."""
from __future__ import annotations

import numpy as np
import pytest
from hydra import compose, initialize

from envs.curriculum import Curriculum
from dynamics.types import UPRIGHT_QUAT


@pytest.fixture
def cfg():
    with initialize(config_path="../configs", version_base=None):
        return compose(config_name="train")


def test_progress_at_zero(cfg) -> None:
    """At step 0, curriculum progress is exactly 0."""
    c = Curriculum(cfg)
    assert c.progress(0) == 0.0


def test_progress_at_anneal_steps(cfg) -> None:
    """At step == anneal_steps, progress is exactly 1.0."""
    c = Curriculum(cfg)
    assert c.progress(cfg.env.curriculum.anneal_steps) == 1.0


def test_progress_clamps_above_full(cfg) -> None:
    """Beyond anneal_steps, progress remains clamped at 1.0 (doesn't
    overshoot — curriculum holds at max difficulty).
    """
    c = Curriculum(cfg)
    assert c.progress(int(cfg.env.curriculum.anneal_steps * 10)) == 1.0


def test_progress_lerp_midpoint(cfg) -> None:
    """At step == anneal_steps / 2, progress is exactly 0.5 (linear lerp)."""
    c = Curriculum(cfg)
    mid = int(cfg.env.curriculum.anneal_steps // 2)
    assert np.isclose(c.progress(mid), 0.5, atol=1e-9)


def test_initial_conditions_at_easy_difficulty(cfg) -> None:
    """At progress=0, lateral offsets are exactly 0, descent velocity is at
    its minimum, altitude is at its minimum, attitude is upright.
    """
    c = Curriculum(cfg)
    rng = np.random.default_rng(seed=42)
    pos, vel, quat, omega = c.sample_initial_conditions(rng, progress=0.0)

    # Lateral offsets zero
    assert pos[0] == 0.0
    assert pos[1] == 0.0
    # Altitude at minimum (NED z = -altitude)
    assert pos[2] == -cfg.env.init_conditions.altitude_min_m
    # Descent velocity at minimum
    assert vel[0] == 0.0
    assert vel[1] == 0.0
    assert vel[2] == cfg.env.init_conditions.descent_velocity_min_mps
    # Upright attitude
    assert np.allclose(quat, UPRIGHT_QUAT)
    # Zero angular rate
    assert np.allclose(omega, 0.0)


def test_initial_conditions_at_full_difficulty(cfg) -> None:
    """At progress=1, sampled values fall within the full configured envelope."""
    c = Curriculum(cfg)
    rng = np.random.default_rng(seed=123)
    pos, vel, _, _ = c.sample_initial_conditions(rng, progress=1.0)

    init = cfg.env.init_conditions
    # Lateral within full envelope
    assert -init.lateral_offset_max_m <= pos[0] <= init.lateral_offset_max_m
    assert -init.lateral_offset_max_m <= pos[1] <= init.lateral_offset_max_m
    # Altitude within full range (z = -altitude, so z in [-max, -min])
    assert -init.altitude_max_m <= pos[2] <= -init.altitude_min_m
    # Descent velocity within full range
    assert init.descent_velocity_min_mps <= vel[2] <= init.descent_velocity_max_mps
