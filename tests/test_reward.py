"""Unit tests for :mod:`envs.reward`.

Covers the control-theoretic reward model: potential-based shaping (PBRS),
the impact-aware terminal outcome, normalisation/clipping, and a regression
test for the original failure mode (an early crash must not outscore a
controlled descent).
"""
from __future__ import annotations

import numpy as np
import pytest
from hydra import compose, initialize

from dynamics.types import IDENTITY_QUAT, UPRIGHT_QUAT, make_state
from envs.reward import (
    landing_cost,
    potential,
    shaping_reward,
    terminal_reward,
    tilt_from_vertical,
)

GAMMA = 0.99


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


# A "landed" state: on the pad, at rest, upright. Zero landing cost.
def _landed_state():
    return _state_at(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), quat=UPRIGHT_QUAT)


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


# --- landing_cost / potential ---------------------------------------------

def test_landing_cost_zero_at_perfect_state(cfg) -> None:
    """On the pad, at rest, upright ⇒ zero landing cost, zero potential."""
    s = _landed_state()
    assert np.isclose(landing_cost(s, cfg), 0.0, atol=1e-12)
    assert np.isclose(potential(s, cfg), 0.0, atol=1e-12)


def test_landing_cost_positive_when_away_from_goal(cfg) -> None:
    """A state with altitude/velocity/tilt errors has strictly positive cost."""
    s = _state_at(pos=(10.0, 0.0, -50.0), vel=(0.0, 0.0, 20.0))
    assert landing_cost(s, cfg) > 0.0
    assert potential(s, cfg) < 0.0


def test_potential_is_clipped(cfg) -> None:
    """Extreme states do not produce unbounded potentials."""
    clip = float(cfg.reward.potential.clip_abs)
    s = _state_at(pos=(1e6, 1e6, -1e6), vel=(1e6, 1e6, 1e6), omega=(1e3, 1e3, 1e3))
    assert abs(potential(s, cfg)) <= clip + 1e-9
    assert np.isfinite(potential(s, cfg))


# --- shaping_reward (PBRS) ------------------------------------------------

def test_shaping_zero_components_at_perfect_state(cfg) -> None:
    """Staying at the perfect state with no action delta and no fuel burn ⇒
    every dense component is ~0 (potential is 0 at both ends)."""
    s = _landed_state()
    total, comps = shaping_reward(
        s, s, np.zeros(3), prev_action=None, prev_fuel_mass=5000.0,
        gamma=GAMMA, cfg=cfg,
    )
    assert np.isclose(comps["shaping"], 0.0, atol=1e-9)
    assert np.isclose(comps["fuel"], 0.0, atol=1e-12)
    assert np.isclose(comps["smoothness"], 0.0, atol=1e-12)
    assert np.isclose(comps["control"], 0.0, atol=1e-12)
    assert np.isclose(total, 0.0, atol=1e-9)


def test_shaping_positive_when_state_improves(cfg) -> None:
    """Moving from a worse state to a better one yields positive shaping."""
    worse = _state_at(pos=(0.0, 0.0, -60.0), vel=(0.0, 0.0, 20.0))
    better = _state_at(pos=(0.0, 0.0, -20.0), vel=(0.0, 0.0, 5.0))
    _, comps = shaping_reward(
        worse, better, np.zeros(3), None, 5000.0, GAMMA, cfg
    )
    assert comps["shaping"] > 0.0


def test_shaping_negative_when_state_degrades(cfg) -> None:
    """Moving from a better state to a worse one yields negative shaping."""
    better = _state_at(pos=(0.0, 0.0, -20.0), vel=(0.0, 0.0, 5.0))
    worse = _state_at(pos=(0.0, 0.0, -60.0), vel=(0.0, 0.0, 20.0))
    _, comps = shaping_reward(
        better, worse, np.zeros(3), None, 5000.0, GAMMA, cfg
    )
    assert comps["shaping"] < 0.0


def test_shaping_fuel_and_control_penalise(cfg) -> None:
    """Fuel burn, action jerk, and control effort are all small penalties."""
    s = _state_at()
    prev = np.zeros(3)
    curr = np.array([1.0, 1.0, 1.0])
    _, comps = shaping_reward(s, s, curr, prev, prev_fuel_mass=5000.0, gamma=GAMMA, cfg=cfg)
    # 0 kg burned here (fuel unchanged), so fuel == 0; smoothness & control < 0.
    assert comps["smoothness"] < 0.0
    assert comps["control"] < 0.0
    s2 = _state_at(fuel=4900.0)
    _, comps2 = shaping_reward(s, s2, curr, prev, prev_fuel_mass=5000.0, gamma=GAMMA, cfg=cfg)
    assert comps2["fuel"] < 0.0


# --- terminal_reward ------------------------------------------------------

def test_terminal_success_beats_any_crash(cfg) -> None:
    """A safe touchdown scores strictly more than any crash."""
    landed = _landed_state()
    gentle_crash = _state_at(vel=(0.0, 0.0, 3.0))
    hard_crash = _state_at(vel=(0.0, 0.0, 40.0), quat=IDENTITY_QUAT, omega=(2.0, 2.0, 2.0))
    success = terminal_reward("success", landed, cfg)
    assert success > terminal_reward("crash", gentle_crash, cfg)
    assert success > terminal_reward("crash", hard_crash, cfg)


def test_terminal_slow_upright_crash_beats_fast_tilted(cfg) -> None:
    """Ordering: slow upright crash > fast tilted/spinning crash."""
    slow_upright = _state_at(vel=(0.0, 0.0, 3.0), quat=UPRIGHT_QUAT)
    fast_tilted = _state_at(vel=(0.0, 0.0, 40.0), quat=IDENTITY_QUAT, omega=(2.0, 2.0, 2.0))
    assert terminal_reward("crash", slow_upright, cfg) > terminal_reward(
        "crash", fast_tilted, cfg
    )


def test_terminal_crash_beats_out_of_bounds(cfg) -> None:
    """A controlled near-miss crash should beat flying out of bounds."""
    slow_upright = _state_at(vel=(0.0, 0.0, 2.5), quat=UPRIGHT_QUAT)
    s = _state_at(pos=(200.0, 0.0, -100.0))
    assert terminal_reward("crash", slow_upright, cfg) > terminal_reward(
        "out_of_bounds", s, cfg
    )


def test_terminal_penalties_are_clipped(cfg) -> None:
    """Extreme crash states cannot exceed the configured terminal clip."""
    clip = float(cfg.reward.terminal.terminal_clip_abs)
    insane = _state_at(vel=(1e4, 1e4, 1e4), quat=IDENTITY_QUAT, omega=(1e3, 1e3, 1e3))
    val = terminal_reward("crash", insane, cfg)
    assert val >= -clip - 1e-6
    assert np.isfinite(val)


def test_terminal_timeout_carries_state_cost(cfg) -> None:
    """A recoverable timeout state beats a chaotic one, but neither beats
    a real landing."""
    near = _state_at(pos=(0.0, 0.0, -5.0), vel=(0.0, 0.0, 1.0))
    chaotic = _state_at(pos=(40.0, 40.0, -200.0), vel=(0.0, 0.0, 30.0), omega=(3.0, 3.0, 3.0))
    assert terminal_reward("timeout", near, cfg) > terminal_reward("timeout", chaotic, cfg)
    assert terminal_reward("success", _landed_state(), cfg) > terminal_reward(
        "timeout", near, cfg
    )


def test_terminal_non_terminal_reasons_return_zero(cfg) -> None:
    """`ongoing` / `reset` aren't terminal events ⇒ return 0."""
    s = _state_at()
    assert terminal_reward("ongoing", s, cfg) == 0.0
    assert terminal_reward("reset", s, cfg) == 0.0


# --- regression: the original failure mode --------------------------------

def test_immediate_crash_does_not_outscore_controlled_descent(cfg) -> None:
    """Regression for the fast-crash bug.

    An immediate hard crash must score worse than a multi-step controlled
    descent that ends in a gentle near-miss. Because PBRS shaping telescopes,
    the descent's accumulated dense reward does not penalise it for taking
    longer — the comparison is driven by terminal landing quality.
    """
    # Path A: one step, then a hard high-speed crash.
    start = _state_at(pos=(0.0, 0.0, -60.0), vel=(0.0, 0.0, 20.0))
    hard_impact = _state_at(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 40.0), quat=IDENTITY_QUAT)
    a_dense, _ = shaping_reward(start, None, np.zeros(3), None, 5000.0, GAMMA, cfg)
    score_a = a_dense + terminal_reward("crash", hard_impact, cfg)

    # Path B: a sequence of improving states ending in a gentle near-miss crash.
    waypoints = [
        _state_at(pos=(0.0, 0.0, -60.0), vel=(0.0, 0.0, 20.0)),
        _state_at(pos=(0.0, 0.0, -45.0), vel=(0.0, 0.0, 14.0)),
        _state_at(pos=(0.0, 0.0, -25.0), vel=(0.0, 0.0, 8.0)),
        _state_at(pos=(0.0, 0.0, -8.0), vel=(0.0, 0.0, 4.0)),
    ]
    gentle_impact = _state_at(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 3.0), quat=UPRIGHT_QUAT)
    score_b = 0.0
    prev_a = None
    for i in range(len(waypoints) - 1):
        d, _ = shaping_reward(
            waypoints[i], waypoints[i + 1], np.zeros(3), prev_a, 5000.0, GAMMA, cfg
        )
        score_b += d
        prev_a = np.zeros(3)
    # final transition into the terminal (gentle) impact
    d, _ = shaping_reward(waypoints[-1], None, np.zeros(3), prev_a, 5000.0, GAMMA, cfg)
    score_b += d + terminal_reward("crash", gentle_impact, cfg)

    assert score_b > score_a, (
        f"controlled descent ({score_b:.2f}) should beat immediate crash ({score_a:.2f})"
    )


def test_reward_finite_under_extreme_states(cfg) -> None:
    """No NaN/inf anywhere in the pipeline for pathological inputs."""
    extreme = _state_at(pos=(1e6, -1e6, -1e6), vel=(1e6, 1e6, -1e6), omega=(1e4, 1e4, 1e4))
    d, comps = shaping_reward(extreme, extreme, np.array([1.0, -1.0, 1.0]), np.zeros(3), 5000.0, GAMMA, cfg)
    assert np.isfinite(d)
    for v in comps.values():
        assert np.isfinite(v)
    for reason in ("success", "crash", "out_of_bounds", "timeout"):
        assert np.isfinite(terminal_reward(reason, extreme, cfg))
