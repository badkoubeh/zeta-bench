"""Unit tests for :mod:`envs.reward`."""
from __future__ import annotations

import numpy as np
import pytest
from hydra import compose, initialize

from dynamics.types import IDENTITY_QUAT, UPRIGHT_QUAT, make_state
from envs.reward import dense_reward, terminal_reward, tilt_from_vertical


@pytest.fixture
def cfg():
    with initialize(config_path="../configs", version_base=None):
        return compose(config_name="train")


def _state_at(
    pos=(0.0, 0.0, 0.0),
    vel=(0.0, 0.0, 0.0),
    quat=None,
    omega=(0.0, 0.0, 0.0),
    fuel=5000.0,
):
    """Build a state with overridable components."""
    if quat is None:
        quat = UPRIGHT_QUAT
    return make_state(
        position_NED=np.asarray(pos, dtype=np.float64),
        velocity_NED=np.asarray(vel, dtype=np.float64),
        quat_wxyz=np.asarray(quat, dtype=np.float64),
        angular_rate_body=np.asarray(omega, dtype=np.float64),
        fuel_mass_kg=float(fuel),
    )


# --- tilt_from_vertical ---------------------------------------------------

def test_tilt_zero_when_upright() -> None:
    """An upright rocket has zero tilt-from-vertical."""
    s = _state_at(quat=UPRIGHT_QUAT)
    assert tilt_from_vertical(s) < 1e-9


def test_tilt_ninety_when_identity_quaternion() -> None:
    """Identity quaternion (body & inertial frames coincide) means body +X
    points along inertial +X (north), which is 90° away from inertial -Z (up).
    """
    s = _state_at(quat=IDENTITY_QUAT)
    assert np.isclose(tilt_from_vertical(s), np.pi / 2, atol=1e-9)


# --- dense_reward ---------------------------------------------------------

def test_dense_reward_zero_at_perfect_state(cfg) -> None:
    """At origin, zero velocity, upright, zero angular rate, full fuel, and
    no action delta — every dense term is zero, so the total is zero."""
    s = _state_at()
    action = np.zeros(3)
    total, components = dense_reward(s, action, prev_action=None, prev_fuel_mass=5000.0, cfg=cfg)
    assert np.isclose(total, 0.0, atol=1e-12)
    for k, v in components.items():
        assert np.isclose(v, 0.0, atol=1e-12), f"component {k} = {v}, expected 0"


def test_dense_reward_distance_penalises_being_far_from_pad(cfg) -> None:
    """Position far from the pad ⇒ negative distance component."""
    s = _state_at(pos=(0.0, 0.0, -100.0))  # 100 m above pad
    _, components = dense_reward(s, np.zeros(3), None, 5000.0, cfg)
    assert components["distance"] < 0.0


def test_dense_reward_fuel_penalises_consumption(cfg) -> None:
    """When fuel decreases between steps, the fuel component is negative."""
    s = _state_at(fuel=4900.0)
    _, components = dense_reward(s, np.zeros(3), None, prev_fuel_mass=5000.0, cfg=cfg)
    # 100 kg burned this step
    assert components["fuel"] < 0.0


def test_dense_reward_smoothness_penalises_jerky_actions(cfg) -> None:
    """Large change between prev_action and action ⇒ negative smoothness component."""
    s = _state_at()
    prev = np.array([0.0, 0.0, 0.0])
    curr = np.array([1.0, 1.0, 1.0])
    _, components = dense_reward(s, curr, prev, 5000.0, cfg)
    assert components["smoothness"] < 0.0


# --- terminal_reward ------------------------------------------------------

def test_terminal_success_matches_config(cfg) -> None:
    assert terminal_reward("success", cfg) == cfg.reward.sparse.success_bonus


def test_terminal_crash_matches_config(cfg) -> None:
    assert terminal_reward("crash", cfg) == cfg.reward.sparse.crash_penalty


def test_terminal_oob_matches_config(cfg) -> None:
    assert terminal_reward("out_of_bounds", cfg) == cfg.reward.sparse.out_of_bounds_penalty


def test_terminal_unrecognised_reason_returns_zero(cfg) -> None:
    """`ongoing` and `timeout` aren't terminal events — return 0 so callers
    can blindly add terminal reward without special-casing.
    """
    assert terminal_reward("ongoing", cfg) == 0.0
    assert terminal_reward("timeout", cfg) == 0.0
