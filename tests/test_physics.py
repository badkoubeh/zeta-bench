"""Physics correctness tests for ModerateFidelityDynamics.

Covers the eight invariants required by the project plan
(``docs/PLAN.md`` Phase 1 Track B): state/action dimensions, gravity-only
freefall kinematics, hover-balance under gravity, throttle bounds, gimbal
torque sign, Tsiolkovsky mass depletion, fuel-empty thrust cutoff, and
quaternion unit-norm preservation under RK4 with renormalisation.

Each test reads like a worked example — expected values are computed in
the docstring so a future reader can replay the arithmetic by hand.

User TODO (per portfolio plan): add 2–3 of your own tests after reviewing
this file. Good candidates: drag-force-vs-velocity-squared scaling,
hover-fuel-burn-time, off-axis thrust translational-rotational coupling,
angular-momentum conservation in the torque-free case.
"""
from __future__ import annotations

import numpy as np

from dynamics.equations_of_motion import (
    G0_TSIOLKOVSKY,
    G_EARTH,
    compute_gimbal_torque_body,
    compute_thrust_force_body,
)
from dynamics.moderate_fidelity import ModerateFidelityDynamics, ModerateFidelityParams
from dynamics.types import (
    ACTION_DIM,
    IDENTITY_QUAT,
    STATE_DIM,
    UPRIGHT_QUAT,
    fuel_mass,
    make_state,
    position,
    quaternion,
    velocity,
)


# --- test fixtures ----------------------------------------------------------

def _make_test_params(**overrides: float | int) -> ModerateFidelityParams:
    """Default Falcon-9-class params (matches configs/env.yaml).

    Overrides let individual tests isolate effects (e.g. ``drag_coefficient=0``
    for a vacuum freefall test).
    """
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


def _initial_state(quat: np.ndarray = IDENTITY_QUAT, fuel_kg: float = 5_000.0) -> np.ndarray:
    """Default at-rest state at the origin."""
    return make_state(
        position_NED=np.zeros(3),
        velocity_NED=np.zeros(3),
        quat_wxyz=quat,
        angular_rate_body=np.zeros(3),
        fuel_mass_kg=fuel_kg,
    )


# --- tests ------------------------------------------------------------------

def test_state_action_dimensions() -> None:
    """Sanity: the published constants match the layout in ``dynamics.types``."""
    assert STATE_DIM == 14
    assert ACTION_DIM == 3
    s = _initial_state()
    assert s.shape == (STATE_DIM,)


def test_zero_thrust_freefall_kinematics() -> None:
    """Drop in vacuum (``Cd=0``), zero thrust, identity attitude.

    Expected (Newtonian, exact in vacuum at constant g):

    .. code-block:: text

        z(t) = z₀ + v_z(0)·t + ½·g·t²   with z₀=0, v_z(0)=0, g=9.81, t=2 s
             = 0 + 0 + 0.5·9.81·4 = 19.62 m
        v_z(t) = v_z(0) + g·t = 0 + 9.81·2 = 19.62 m/s

    NED Z+ is down, so positive z = falling toward the pad.
    """
    params = _make_test_params(drag_coefficient=0.0)
    dyn = ModerateFidelityDynamics(params)
    s = _initial_state()

    dt = 0.02  # 50 Hz control
    for _ in range(100):  # 2.0 s simulated
        s = dyn.step(s, np.zeros(3), dt)

    t = 2.0
    expected_z = 0.5 * G_EARTH * t * t  # 19.62
    expected_vz = G_EARTH * t  # 19.62
    assert np.isclose(position(s)[2], expected_z, atol=1e-3), \
        f"z={position(s)[2]}, expected {expected_z}"
    assert np.isclose(velocity(s)[2], expected_vz, atol=1e-3), \
        f"v_z={velocity(s)[2]}, expected {expected_vz}"


def test_hover_balance() -> None:
    """Vertical attitude, throttle that exactly balances gravity, zero gimbal.

    Setup:
        - Attitude = UPRIGHT_QUAT (body +X = inertial −Z, i.e. nose up).
        - Total mass = dry + fuel = 25 000 + 5 000 = 30 000 kg.
        - Required thrust = m·g = 30 000·9.81 = 294 300 N.
        - Required throttle = 294 300 / 845 000 ≈ 0.3483.

    But the engine's minimum sustainable throttle is 0.4, so the required
    throttle is *below* min — the engine can't actually hover this vehicle
    at full fuel. We make this test physically clean by *raising* the
    weight (heavier fuel load) until the hover throttle lies inside the
    allowed range. Use 30 000 kg fuel → total 55 000 kg → req throttle
    = 55 000·9.81 / 845 000 ≈ 0.6385 (well inside [0.4, 1.0]).

    After one control tick: vertical acceleration should be < 1e-2 m/s²
    (RK4 + small drag still contributes a tiny non-zero acceleration
    because at zero velocity drag is zero, but the next-step velocity is
    small so drag stays small). We assert tight balance on vertical
    velocity after 0.5 s of hover.
    """
    params = _make_test_params(drag_coefficient=0.0)  # isolate gravity vs thrust
    dyn = ModerateFidelityDynamics(params)

    total_mass = params.dry_mass_kg + 30_000.0  # heavier fuel for valid throttle
    required_throttle = (total_mass * G_EARTH) / params.max_thrust_N
    assert params.throttle_min <= required_throttle <= params.throttle_max, \
        f"hover throttle {required_throttle} outside engine limits — fixture bug"

    s = _initial_state(quat=UPRIGHT_QUAT, fuel_kg=30_000.0)
    action = np.array([required_throttle, 0.0, 0.0])

    dt = 0.02
    for _ in range(25):  # 0.5 s of hover
        s = dyn.step(s, action, dt)

    # Mass changed (fuel burned), so required throttle drifts. After 0.5 s of
    # burn at ~300 kN, fuel burned ≈ 300 000 · 0.5 / (282 · 9.80665) ≈ 54 kg —
    # negligible fraction. Acceleration drift should be tiny.
    assert abs(velocity(s)[2]) < 5e-2, f"hover velocity drift {velocity(s)[2]} too large"


def test_throttle_bounds() -> None:
    """Throttle saturates to ``[throttle_min, throttle_max]`` when commanded.

    Spot-checks:
        - throttle_cmd = 0  → thrust magnitude = 0
        - throttle_cmd = 0.5 → thrust = 0.5 · max_thrust_N (inside range)
        - throttle_cmd = 1.5 → thrust = max_thrust_N (saturated up)
        - throttle_cmd negative → thrust = 0 (engine off)
    """
    params = _make_test_params()

    def thrust_magnitude(throttle_cmd: float) -> float:
        action = np.array([throttle_cmd, 0.0, 0.0])
        F = compute_thrust_force_body(action, params, fuel_mass_kg=1000.0)
        return float(np.linalg.norm(F))

    assert thrust_magnitude(0.0) == 0.0
    assert np.isclose(thrust_magnitude(0.5), 0.5 * params.max_thrust_N)
    assert np.isclose(thrust_magnitude(1.5), params.max_thrust_N)
    assert thrust_magnitude(-0.2) == 0.0


def test_gimbal_pitch_produces_pitch_torque() -> None:
    """Non-zero gimbal pitch generates a body-frame pitch torque with the
    expected sign and magnitude.

    With ``gimbal_pitch_cmd = +1`` (full positive deflection):
        ε_p = +gimbal_max_rad
        F_x_body = T·cos(ε_p)·1
        F_y_body = 0
        F_z_body = −T·sin(ε_p)
        τ_body = (0, L·F_z, −L·F_y) = (0, −L·T·sin(ε_p), 0)

    For positive ε_p, ``sin(ε_p) > 0``, so ``τ_y = −L·T·sin(ε_p) < 0``.
    A negative τ_y in the FRD body frame rotates the rocket nose-up
    (positive pitch in the rocket sense), which is what you'd expect: tilt
    the engine such that thrust pushes the engine end toward the
    +Z_body side → rocket rotates so the nose goes the opposite way.

    Roll torque must be exactly zero (no roll authority).
    """
    params = _make_test_params()
    F = compute_thrust_force_body(
        action=np.array([1.0, 1.0, 0.0]),  # full throttle, full pitch gimbal
        params=params,
        fuel_mass_kg=1000.0,
    )
    tau = compute_gimbal_torque_body(F, params.engine_lever_arm_m)

    # Roll torque is identically zero
    assert tau[0] == 0.0

    # Pitch torque has the expected sign and magnitude
    T = params.max_thrust_N
    eps_p = params.gimbal_max_rad
    expected_tau_y = -params.engine_lever_arm_m * T * np.sin(eps_p)
    assert np.isclose(tau[1], expected_tau_y, rtol=1e-9), \
        f"τ_y = {tau[1]}, expected {expected_tau_y}"

    # Yaw torque should be zero (zero yaw gimbal command)
    assert np.isclose(tau[2], 0.0, atol=1e-9)


def test_fuel_depletion_tsiolkovsky() -> None:
    """Constant thrust → fuel depletes linearly at the rate ``T / (Isp·g₀)``.

    With throttle = 1.0, gimbal = 0:
        T = max_thrust_N = 845 000 N
        Isp = 282 s, g₀ = 9.80665
        ṁ = −T / (Isp·g₀) = −845 000 / (282 · 9.80665) ≈ −305.49 kg/s

    After 0.5 s, fuel consumed ≈ 152.74 kg. Starting fuel = 5 000 kg →
    remaining ≈ 4 847.26 kg.
    """
    params = _make_test_params(drag_coefficient=0.0)  # drag wouldn't affect mass anyway
    dyn = ModerateFidelityDynamics(params)
    s = _initial_state(quat=UPRIGHT_QUAT)
    action = np.array([1.0, 0.0, 0.0])  # full throttle

    dt = 0.02
    for _ in range(25):  # 0.5 s
        s = dyn.step(s, action, dt)

    expected_burn_rate = params.max_thrust_N / (params.isp_s * G0_TSIOLKOVSKY)  # ≈ 305.49
    expected_consumed = expected_burn_rate * 0.5  # ≈ 152.74
    expected_remaining = 5_000.0 - expected_consumed
    assert np.isclose(fuel_mass(s), expected_remaining, rtol=1e-4), \
        f"fuel = {fuel_mass(s)}, expected {expected_remaining}"


def test_fuel_empty_disables_thrust() -> None:
    """With fuel_mass = 0, commanded thrust produces no force."""
    params = _make_test_params()
    F = compute_thrust_force_body(
        action=np.array([1.0, 0.0, 0.0]),  # commanded full throttle
        params=params,
        fuel_mass_kg=0.0,  # but no fuel
    )
    assert np.allclose(F, 0.0)


def test_quaternion_unit_norm_after_step() -> None:
    """After a step with non-trivial angular velocity, ‖q‖ stays at 1.

    Starts with identity attitude and an angular-velocity vector with all
    three body components non-zero, so q̇ has all four components non-zero
    and the integrator must renormalise to avoid drift.
    """
    params = _make_test_params(drag_coefficient=0.0)
    dyn = ModerateFidelityDynamics(params)
    s = make_state(
        position_NED=np.zeros(3),
        velocity_NED=np.zeros(3),
        quat_wxyz=IDENTITY_QUAT,
        angular_rate_body=np.array([0.3, 0.5, 0.7]),  # rad/s, body frame
        fuel_mass_kg=5_000.0,
    )

    for _ in range(50):  # 1 s of free rotation
        s = dyn.step(s, np.zeros(3), 0.02)

    q_norm = float(np.linalg.norm(quaternion(s)))
    assert abs(q_norm - 1.0) < 1e-10, f"‖q‖ drifted to {q_norm}"
