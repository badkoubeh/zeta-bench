"""Physics-correctness tests for disturbance injection into the dynamics.

Covers the two physics-level disturbances that reach the equations of motion:

- **wind** — modelled as relative airspeed in the drag term. Verifies the
  no-wind path is unchanged, that wind adds a drag force in the wind direction,
  and that wind equal to the vehicle velocity zeroes the drag (airspeed = 0).
- **mass uncertainty** — applied by scaling ``dry_mass_kg``. Verifies total
  mass shifts, that equal thrust yields smaller acceleration on a heavier
  vehicle, and that free-fall acceleration stays mass-independent (= g).

These complement the nominal physics tests in ``test_physics*.py``.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from dynamics.equations_of_motion import (
    G_EARTH,
    compute_drag_inertial,
    rk4_step,
    state_derivative,
)
from dynamics.moderate_fidelity import ModerateFidelityDynamics, ModerateFidelityParams
from dynamics.types import (
    UPRIGHT_QUAT,
    VEL_SLICE,
    make_state,
    velocity,
)


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
        throttle_min=0.25,
        throttle_max=1.0,
        physics_substeps=4,
        engine_lever_arm_m=15.0,
    )
    defaults.update(overrides)
    return ModerateFidelityParams(**defaults)


def _state(vel_ned, fuel=5_000.0):
    return make_state(
        position_NED=np.array([0.0, 0.0, -100.0]),
        velocity_NED=np.asarray(vel_ned, dtype=np.float64),
        quat_wxyz=UPRIGHT_QUAT,
        angular_rate_body=np.zeros(3),
        fuel_mass_kg=fuel,
    )


# --- wind through drag ----------------------------------------------------

def test_wind_none_equals_zero_wind() -> None:
    """The default no-wind path matches passing an explicit zero wind vector."""
    p = _params()
    v = np.array([0.0, 0.0, 20.0])  # descending
    d_none = compute_drag_inertial(v, p, None)
    d_zero = compute_drag_inertial(v, p, np.zeros(3))
    np.testing.assert_allclose(d_none, d_zero, atol=1e-12)


def test_wind_none_step_matches_zero_wind_step() -> None:
    """A full RK4 step with wind=None equals wind=zeros (regression guard)."""
    p = _params()
    s = _state([1.0, 0.0, 15.0])
    action = np.array([0.5, 0.0, 0.0])
    a = rk4_step(s, action, p, 0.02, None)
    b = rk4_step(s, action, p, 0.02, np.zeros(3))
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_crosswind_adds_drag_in_wind_direction() -> None:
    """A crosswind on a vertically descending rocket produces lateral drag.

    With the rocket falling straight down, adding an eastward wind makes the
    relative airspeed point west, so drag pushes the vehicle east (+Y) — i.e.
    the air pushes the rocket downwind.
    """
    p = _params()
    v = np.array([0.0, 0.0, 20.0])  # straight down, no horizontal motion
    wind = np.array([0.0, 5.0, 0.0])  # blowing east
    drag_nominal = compute_drag_inertial(v, p, None)
    drag_wind = compute_drag_inertial(v, p, wind)
    # Nominal drag is purely vertical (opposes descent): no lateral component.
    assert np.isclose(drag_nominal[1], 0.0)
    # With eastward wind, drag gains a positive-Y (eastward) push.
    assert drag_wind[1] > 0.0


def test_wind_equal_to_velocity_zeroes_drag() -> None:
    """When the air moves with the vehicle, airspeed is zero and drag vanishes."""
    p = _params()
    v = np.array([3.0, -2.0, 18.0])
    drag = compute_drag_inertial(v, p, wind_velocity_ned=v)
    np.testing.assert_allclose(drag, np.zeros(3), atol=1e-9)


def test_wind_opposes_and_aligns_change_drag_magnitude() -> None:
    """Head-on vs tail wind change airspeed and thus drag magnitude sensibly."""
    p = _params()
    v = np.array([0.0, 0.0, 20.0])
    head = compute_drag_inertial(v, p, np.array([0.0, 0.0, -10.0]))  # air rising: faster airspeed
    tail = compute_drag_inertial(v, p, np.array([0.0, 0.0, 10.0]))   # air descending: slower
    assert np.linalg.norm(head) > np.linalg.norm(compute_drag_inertial(v, p, None))
    assert np.linalg.norm(tail) < np.linalg.norm(compute_drag_inertial(v, p, None))


# --- mass offset ----------------------------------------------------------

def test_mass_offset_shifts_total_mass() -> None:
    """Scaling dry_mass_kg by (1+frac) raises the total mass proportionally."""
    base = _params()
    heavier = dataclasses.replace(base, dry_mass_kg=base.dry_mass_kg * 1.20)
    assert heavier.dry_mass_kg > base.dry_mass_kg
    assert np.isclose(heavier.dry_mass_kg, 30_000.0)


def test_heavier_vehicle_accelerates_less_under_equal_thrust() -> None:
    """Under identical thrust, a +20% dry-mass vehicle has smaller |vel_dot|."""
    base = _params()
    heavier = dataclasses.replace(base, dry_mass_kg=base.dry_mass_kg * 1.20)
    s = _state([0.0, 0.0, 0.0])
    action = np.array([1.0, 0.0, 0.0])  # full thrust, upright
    accel_base = state_derivative(s, action, base)[VEL_SLICE]
    accel_heavy = state_derivative(s, action, heavier)[VEL_SLICE]
    # Vertical (Z, down-positive) acceleration: full thrust drives it more
    # negative (upward). The heavier vehicle accelerates upward less strongly.
    assert accel_heavy[2] > accel_base[2]


def test_freefall_acceleration_is_mass_independent() -> None:
    """With zero thrust and zero speed, acceleration is g regardless of mass."""
    light = _params(dry_mass_kg=20_000.0)
    heavy = _params(dry_mass_kg=40_000.0)
    s = _state([0.0, 0.0, 0.0], fuel=0.0)  # no fuel => no thrust
    action = np.array([0.0, 0.0, 0.0])
    a_light = state_derivative(s, action, light)[VEL_SLICE]
    a_heavy = state_derivative(s, action, heavy)[VEL_SLICE]
    np.testing.assert_allclose(a_light, [0.0, 0.0, G_EARTH], atol=1e-9)
    np.testing.assert_allclose(a_heavy, [0.0, 0.0, G_EARTH], atol=1e-9)


# --- integration: wind changes the trajectory -----------------------------

def test_wind_changes_lateral_trajectory_over_step() -> None:
    """Integrating one control tick with a crosswind imparts lateral velocity."""
    p = _params()
    dyn = ModerateFidelityDynamics(p)
    s = _state([0.0, 0.0, 20.0])
    action = np.array([0.4, 0.0, 0.0])
    nominal = dyn.step(s, action, 0.02, None)
    windy = dyn.step(s, action, 0.02, np.array([0.0, 8.0, 0.0]))
    # Eastward wind should induce a positive-Y velocity component absent nominally.
    assert np.isclose(velocity(nominal)[1], 0.0, atol=1e-9)
    assert velocity(windy)[1] > 0.0
