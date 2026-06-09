"""Property-based physics tests using Hypothesis.

Each property test generates ~50 random inputs from a constrained strategy
and asserts an invariant that must hold under all valid inputs. These
complement the canonical-input tests in ``test_physics.py`` and the
targeted edge-case tests in ``test_physics_edge_cases.py``.

Hypothesis strategies are explicit and bounded — physical ranges only, no
NaN/Inf, no degenerate quaternions. ``max_examples`` is set per-test to
keep total runtime under ~1 s.

Reading this file should communicate: "the implementation upholds the
following invariants under arbitrary valid inputs, not just hand-picked
ones."
"""
from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from dynamics.equations_of_motion import (
    compute_drag_inertial,
    compute_thrust_force_body,
)
from dynamics.moderate_fidelity import ModerateFidelityDynamics, ModerateFidelityParams
from dynamics.types import IDENTITY_QUAT, fuel_mass, make_state, quaternion


# --- shared fixture -------------------------------------------------------

def _params(**overrides: float | int) -> ModerateFidelityParams:
    defaults: dict[str, float | int] = dict(
        dry_mass_kg=25_000.0,
        initial_fuel_kg=5_000.0,
        max_thrust_N=845_000.0,
        isp_s=282.0,
        drag_coefficient=0.75,
        reference_area_m2=10.5,
        inertia_xx=2.5e6,
        inertia_yy=2.5e6,
        inertia_zz=1.2e5,
        gimbal_max_rad=0.0873,
        throttle_min=0.4,
        throttle_max=1.0,
        physics_substeps=4,
        engine_lever_arm_m=15.0,
    )
    defaults.update(overrides)
    return ModerateFidelityParams(**defaults)


# --- strategies -----------------------------------------------------------

@st.composite
def unit_quaternions(draw: st.DrawFn) -> np.ndarray:
    """Uniformly sample a 4-vector and normalise — produces a unit quaternion.

    ``assume`` filters out near-zero raw vectors that would normalise into
    NaN. Hypothesis will retry with a different draw.
    """
    raw = np.array(
        [
            draw(st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False))
            for _ in range(4)
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(raw))
    assume(norm > 1e-6)
    return raw / norm


# Body-frame angular rate (rad/s), bounded for physical sanity
body_omegas = st.lists(
    st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
    min_size=3,
    max_size=3,
).map(lambda lst: np.array(lst, dtype=np.float64))


# 3-D velocity in m/s, up to 100 m/s magnitude per axis
velocities = st.lists(
    st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
    min_size=3,
    max_size=3,
).map(lambda lst: np.array(lst, dtype=np.float64))


# Actions in the env's Box bounds
actions = st.tuples(
    st.floats(0.0, 1.0, allow_nan=False),
    st.floats(-1.0, 1.0, allow_nan=False),
    st.floats(-1.0, 1.0, allow_nan=False),
).map(lambda t: np.array(t, dtype=np.float64))


# Throttles in the active range [throttle_min, throttle_max]
throttles_in_active_range = st.floats(0.4, 1.0, allow_nan=False)


# --- property: quaternion norm preservation -------------------------------

@given(q=unit_quaternions(), omega=body_omegas)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property_quaternion_norm_preserved_under_step(
    q: np.ndarray, omega: np.ndarray
) -> None:
    """Invariant: for any unit quaternion and any body-frame ω in a
    reasonable range, one ``dyn.step`` keeps ‖q‖ within 1e-9 of 1.

    Tests the post-integration quaternion renormalisation in :func:`rk4_step`.
    """
    params = _params(drag_coefficient=0.0)  # isolate rotational dynamics
    dyn = ModerateFidelityDynamics(params)
    s = make_state(
        position_NED=np.zeros(3),
        velocity_NED=np.zeros(3),
        quat_wxyz=q,
        angular_rate_body=omega,
        fuel_mass_kg=5_000.0,
    )
    s_next = dyn.step(s, np.zeros(3), dt=0.02)
    assert abs(float(np.linalg.norm(quaternion(s_next))) - 1.0) < 1e-9


# --- property: fuel non-negativity ----------------------------------------

@given(action_sequence=st.lists(actions, min_size=10, max_size=40))
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property_fuel_non_negative_over_action_sequence(
    action_sequence: list[np.ndarray],
) -> None:
    """Invariant: across any sequence of valid actions, fuel mass stays
    non-negative. ``rk4_step`` clamps fuel ≥ 0; this exercises arbitrary
    action histories to catch any path where the clamp could fail.

    Starts with a small fuel load (100 kg) so most action sequences deplete
    it within the test — actively exercising the clamp branch.
    """
    params = _params(initial_fuel_kg=100.0)
    dyn = ModerateFidelityDynamics(params)
    s = make_state(
        position_NED=np.zeros(3),
        velocity_NED=np.zeros(3),
        quat_wxyz=IDENTITY_QUAT,
        angular_rate_body=np.zeros(3),
        fuel_mass_kg=100.0,
    )
    for action in action_sequence:
        s = dyn.step(s, action, dt=0.02)
        assert fuel_mass(s) >= 0.0


# --- property: thrust monotonic in throttle -------------------------------

@given(
    throttle_a=throttles_in_active_range,
    throttle_b=throttles_in_active_range,
)
@settings(max_examples=50, deadline=None)
def test_property_thrust_monotonic_in_throttle(
    throttle_a: float, throttle_b: float
) -> None:
    """Invariant: for fixed gimbal (zero), ‖F_thrust‖ is monotonically
    non-decreasing in throttle inside the active range. Within
    [throttle_min, throttle_max] the clamp is a no-op, so monotonicity
    must hold exactly.
    """
    params = _params()
    F_a = compute_thrust_force_body(
        np.array([throttle_a, 0.0, 0.0]), params, fuel_mass_kg=1000.0
    )
    F_b = compute_thrust_force_body(
        np.array([throttle_b, 0.0, 0.0]), params, fuel_mass_kg=1000.0
    )
    mag_a = float(np.linalg.norm(F_a))
    mag_b = float(np.linalg.norm(F_b))
    if throttle_a < throttle_b:
        assert mag_a <= mag_b
    elif throttle_a > throttle_b:
        assert mag_a >= mag_b


# --- property: drag opposes velocity --------------------------------------

@given(v=velocities)
@settings(max_examples=50, deadline=None)
def test_property_drag_opposes_velocity(v: np.ndarray) -> None:
    """Invariant: drag force is antiparallel to velocity (or zero at rest).

    Equivalent to ``F_drag × v = 0`` AND ``F_drag · v ≤ 0``. Tolerance on
    the cross product is scaled by speed because both sides scale with
    speed and float precision loses bits for large numbers.
    """
    params = _params()
    F_drag = compute_drag_inertial(v, params)
    speed = float(np.linalg.norm(v))
    if speed < 1e-9:
        assert np.allclose(F_drag, 0.0)
    else:
        # Cross product near zero (parallel) — tolerance grows with speed²
        # (drag magnitude scales as v², so absolute floating-point error
        # grows with v²·v = v³ in the worst case)
        assert np.allclose(np.cross(F_drag, v), 0.0, atol=1e-6 * speed**3 + 1e-9)
        # Dot product non-positive (antiparallel, never aligned with v)
        assert float(np.dot(F_drag, v)) <= 0.0
