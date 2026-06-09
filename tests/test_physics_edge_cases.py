"""Targeted edge-case tests filling visible gaps in branch coverage.

Each test exercises a specific branch / function that the canonical-input
tests in ``test_physics.py`` did not directly hit. Organised one-section-
per-function so a coverage gap is easy to localise.

Generated to close gaps identified in the test-coverage-hardening plan:
- ``compute_drag_inertial`` non-zero-velocity branch (v² scaling, direction)
- ``compute_angular_acceleration`` gyroscopic cross-coupling term
- ``compute_thrust_force_body`` / ``clamp_throttle`` boundary branches
- ``ModerateFidelityDynamics.get_params``
- ``make_state`` round-trip with accessors
"""
from __future__ import annotations

import numpy as np

from dynamics.equations_of_motion import (
    clamp_throttle,
    compute_angular_acceleration,
    compute_drag_inertial,
    compute_thrust_force_body,
)
from dynamics.moderate_fidelity import ModerateFidelityDynamics, ModerateFidelityParams
from dynamics.types import (
    angular_rate,
    fuel_mass,
    make_state,
    position,
    quaternion,
    velocity,
)


# --- shared fixture --------------------------------------------------------

def _make_test_params(**overrides: float | int) -> ModerateFidelityParams:
    """Default Falcon-9-class params (matches configs/env.yaml)."""
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


# --- compute_drag_inertial -------------------------------------------------

def test_drag_scales_as_velocity_squared() -> None:
    """Drag magnitude scales as ‖v‖²: doubling velocity quadruples drag.

    With identical direction and the same vehicle, F_drag = ½·ρ·v²·Cd·A so
    the ratio of drag magnitudes is exactly the ratio of v² values.
    """
    params = _make_test_params()
    v1 = np.array([10.0, 0.0, 0.0])
    v2 = np.array([20.0, 0.0, 0.0])
    F1 = compute_drag_inertial(v1, params)
    F2 = compute_drag_inertial(v2, params)
    ratio = float(np.linalg.norm(F2) / np.linalg.norm(F1))
    assert np.isclose(ratio, 4.0, rtol=1e-9), f"ratio={ratio}, expected 4.0"


def test_drag_opposes_velocity_in_arbitrary_direction() -> None:
    """For any non-zero v, drag is antiparallel to v.

    Equivalent to ``F_drag × v = 0`` (parallel/antiparallel) AND
    ``F_drag · v < 0`` (antiparallel). Tested in 3-D with an arbitrary
    velocity direction.
    """
    params = _make_test_params()
    v = np.array([3.0, -4.0, 5.0])
    F = compute_drag_inertial(v, params)
    assert np.allclose(np.cross(F, v), 0.0, atol=1e-9)
    assert float(np.dot(F, v)) < 0.0


# --- compute_angular_acceleration ------------------------------------------

def test_gyroscopic_coupling_with_asymmetric_inertia() -> None:
    """With zero torque, asymmetric inertia, and non-zero ω, ω̇ is determined
    by the ``ω × (I·ω)`` cross-coupling term.

    With ω = (1, 1, 1) and I = diag(1, 2, 3):

    .. code-block:: text

        ω̇_x = (0 − (I_zz − I_yy)·ω_y·ω_z) / I_xx = −(3 − 2)·1·1 / 1 = −1
        ω̇_y = (0 − (I_xx − I_zz)·ω_z·ω_x) / I_yy = −(1 − 3)·1·1 / 2 = +1
        ω̇_z = (0 − (I_yy − I_xx)·ω_x·ω_y) / I_zz = −(2 − 1)·1·1 / 3 = −1/3

    A zero-torque rotation about a non-principal axis MUST produce non-zero
    angular acceleration if Euler's equations are implemented correctly.
    """
    omega = np.array([1.0, 1.0, 1.0])
    torque = np.zeros(3)
    inertia = np.array([1.0, 2.0, 3.0])
    omega_dot = compute_angular_acceleration(omega, torque, inertia)
    expected = np.array([-1.0, 1.0, -1.0 / 3.0])
    assert np.allclose(omega_dot, expected, atol=1e-12)


# --- clamp_throttle / compute_thrust_force_body ----------------------------

def test_throttle_clamps_up_to_min() -> None:
    """When ``0 < throttle_cmd < throttle_min``, ``clamp_throttle`` raises it
    to ``throttle_min`` (engines below idle aren't sustainable, so we treat
    the deep-throttle command as "go to min sustainable" — see docstring of
    clamp_throttle for the alternative semantics).
    """
    assert clamp_throttle(0.2, 0.4, 1.0, fuel_mass_kg=1000.0) == 0.4


def test_throttle_saturates_at_max() -> None:
    """When ``throttle_cmd > throttle_max``, ``clamp_throttle`` saturates it
    at ``throttle_max``.
    """
    assert clamp_throttle(1.5, 0.4, 1.0, fuel_mass_kg=1000.0) == 1.0


def test_thrust_zero_when_negative_throttle_commanded() -> None:
    """Negative throttle command produces zero thrust (engine off semantics)."""
    params = _make_test_params()
    F = compute_thrust_force_body(
        action=np.array([-0.5, 0.0, 0.0]),
        params=params,
        fuel_mass_kg=1000.0,
    )
    assert np.allclose(F, 0.0)


# --- ModerateFidelityDynamics.get_params -----------------------------------

def test_get_params_returns_dataclass_field_values() -> None:
    """``get_params`` returns a 14-element float64 ndarray whose elements
    match the field order of :class:`ModerateFidelityParams`.
    """
    params = _make_test_params()
    dyn = ModerateFidelityDynamics(params)
    p = dyn.get_params()

    assert p.shape == (14,)
    assert p.dtype == np.float64

    expected = np.array(
        [
            params.dry_mass_kg,
            params.initial_fuel_kg,
            params.max_thrust_N,
            params.isp_s,
            params.drag_coefficient,
            params.reference_area_m2,
            params.inertia_xx,
            params.inertia_yy,
            params.inertia_zz,
            params.gimbal_max_rad,
            params.throttle_min,
            params.throttle_max,
            float(params.physics_substeps),
            params.engine_lever_arm_m,
        ]
    )
    assert np.allclose(p, expected)


# --- make_state ------------------------------------------------------------

def test_make_state_round_trips_through_accessors() -> None:
    """``make_state`` correctly assembles a 14-vector with the slot layout
    documented in ``dynamics.types``. Each accessor returns the original
    named component.
    """
    pos = np.array([1.0, 2.0, 3.0])
    vel = np.array([4.0, 5.0, 6.0])
    # Arbitrary unit quaternion (rotation about an axis); normalise to be safe
    quat_raw = np.array([0.5, 0.5, 0.5, 0.5])
    quat = quat_raw / np.linalg.norm(quat_raw)
    omega = np.array([0.1, 0.2, 0.3])
    fuel = 4_500.0

    s = make_state(pos, vel, quat, omega, fuel)

    assert s.shape == (14,)
    assert s.dtype == np.float64
    assert np.allclose(position(s), pos)
    assert np.allclose(velocity(s), vel)
    assert np.allclose(quaternion(s), quat)
    assert np.allclose(angular_rate(s), omega)
    assert fuel_mass(s) == fuel
