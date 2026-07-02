"""6-DOF rigid-body equations of motion for the rocket-landing project.

All functions in this module are **pure**: they read state + action + params
and return derivatives or rotated vectors. The integrator :func:`rk4_step`
is also pure — it does not mutate its inputs.

Reading order
-------------
The functions are arranged so a reviewer can follow the derivation top-to-
bottom: rotation helpers → force / torque computations → :func:`state_derivative`
(which composes them) → :func:`rk4_step` (which integrates).

Frame conventions (also stated in :mod:`dynamics.types`)
--------------------------------------------------------
- Inertial: NED — X = North, Y = East, Z = Down. Gravity g = +9.81 ẑ.
- Body: FRD — X = Forward (nose), Y = Right, Z = Down (toward belly).
- Attitude q = (w, x, y, z), Hamilton convention, unit-norm.

References
----------
- Translational: Newton's second law in inertial frame.
- Rotational: Euler's equations for a rigid body with principal-axes
  inertia tensor. Greenwood, *Advanced Dynamics* §8.3.
- Quaternion kinematics: ``q̇ = ½ q ⊗ ω̃`` with ``ω̃ = (0, ωₓ, ωᵧ, ω_z)``.
  Wertz, *Spacecraft Attitude Determination and Control* §16.1.
- Mass depletion (Tsiolkovsky surrogate): ``ṁ = −T / (Iₛₚ · g₀)``.
- RK4 derivation: any numerical-methods textbook (e.g. Hairer & Wanner).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from dynamics.types import (
    Action,
    State,
    angular_rate,
    fuel_mass,
    quaternion,
    velocity,
)

# --- physical constants -----------------------------------------------------
# G0 is the *standard gravity* used in the Tsiolkovsky / specific-impulse
# relation (ṁ = T / (Isp · g₀)). It is a definitional constant, NOT a local
# gravitational acceleration.
G0_TSIOLKOVSKY: float = 9.80665  # m/s²

# Local gravitational acceleration in the gravity-force term. Constant at
# sea-level value because moderate fidelity ignores altitude variation
# (drop heights < 600 m make the variation < 0.02%).
G_EARTH: float = 9.81  # m/s²

# Sea-level air density (ISA). Constant because moderate fidelity ignores
# altitude variation (a real model would interpolate over ρ(h)).
RHO_AIR_SEA_LEVEL: float = 1.225  # kg/m³


# --- rotation helpers -------------------------------------------------------

def quat_rotate_body_to_inertial(
    q: NDArray[np.float64],
    v_body: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Rotate a 3-vector from body frame to inertial frame.

    Hamilton convention: ``q = (w, x, y, z)``, body-to-inertial, assumed
    unit norm.

    Implements ``v_inertial = R(q) · v_body`` with the standard mapping

    .. code-block:: text

        R = [ 1-2(y²+z²),  2(xy-wz),    2(xz+wy)   ]
            [ 2(xy+wz),    1-2(x²+z²),  2(yz-wx)   ]
            [ 2(xz-wy),    2(yz+wx),    1-2(x²+y²) ]

    Parameters
    ----------
    q : NDArray[(4,), float64]
        Attitude quaternion (w, x, y, z), unit-norm.
    v_body : NDArray[(3,), float64]
        Vector expressed in the body frame.

    Returns
    -------
    NDArray[(3,), float64]
        Same vector expressed in the inertial frame.
    """
    w, x, y, z = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
            [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
        ]
    )
    return R @ v_body


def quat_to_euler(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a unit quaternion to aerospace ZYX Euler angles (roll, pitch, yaw).

    Hamilton convention, body-to-inertial quaternion `q = (w, x, y, z)`.

    .. code-block:: text

        roll  = atan2( 2(w·x + y·z),   1 − 2(x² + y²) )
        pitch = asin (  2(w·y − z·x) )
        yaw   = atan2( 2(w·z + x·y),   1 − 2(y² + z²) )

    Pitch is clamped to ±π/2 (the singularity range). Standard aerospace
    decomposition; suffers gimbal lock at pitch = ±π/2 (which is the
    "perfectly nose-up" attitude for our tail-firing rocket — accept this
    in moderate fidelity, the observer's tilt-from-vertical check uses
    a direct quaternion→axis projection instead).

    Returns
    -------
    NDArray[(3,), float64]
        Euler angles (roll, pitch, yaw) in radians.
    """
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = float(np.arctan2(sinr_cosp, cosr_cosp))

    sinp = 2.0 * (w * y - z * x)
    pitch = float(np.arcsin(np.clip(sinp, -1.0, 1.0)))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = float(np.arctan2(siny_cosp, cosy_cosp))

    return np.array([roll, pitch, yaw], dtype=np.float64)


def quat_kinematics_dot(
    q: NDArray[np.float64],
    omega_body: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the time-derivative of the attitude quaternion.

    Hamilton convention: ``q̇ = ½ q ⊗ ω̃`` where ``ω̃ = (0, ωₓ, ωᵧ, ω_z)`` is
    the pure-imaginary quaternion encoding of the body-frame angular
    velocity.

    Working out the Hamilton quaternion product ``q ⊗ (0, p, q_y, r)``:

    .. code-block:: text

        q ⊗ ω̃ = ( -x·p - y·q_y - z·r,
                   w·p + y·r - z·q_y,
                   w·q_y - x·r + z·p,
                   w·r + x·q_y - y·p )

    Multiply by ½ to get q̇.

    Notes
    -----
    Integrating q̇ over a finite step drifts q off the unit sphere by
    O(dt²). The integrator (:func:`rk4_step`) renormalises after each step.
    """
    w, x, y, z = q
    p, q_y, r = omega_body  # roll, pitch, yaw rates in body frame
    return 0.5 * np.array(
        [
            -x * p - y * q_y - z * r,
             w * p + y * r   - z * q_y,
             w * q_y - x * r + z * p,
             w * r   + x * q_y - y * p,
        ]
    )


# --- force / torque computations -------------------------------------------

def clamp_throttle(
    throttle_cmd: float,
    throttle_min: float,
    throttle_max: float,
    fuel_mass_kg: float,
) -> float:
    """Apply throttle saturation and fuel-empty cutoff.

    Real liquid-fuel engines cannot sustain combustion below a minimum
    throttle (typically 40-60% of nominal). Above 100% they're saturated
    by hardware limits. With no fuel, thrust is zero regardless of command.

    The "deep-throttle" range below ``throttle_min`` is mapped to zero
    here (engine is off), not to ``throttle_min`` (which would be the
    "min sustainable thrust" semantics). Either choice is defensible —
    we model OFF/ON, not idle.
    """
    if fuel_mass_kg <= 0.0:
        return 0.0
    if throttle_cmd <= 0.0:
        return 0.0
    return float(np.clip(throttle_cmd, throttle_min, throttle_max))


def compute_thrust_force_body(
    action: Action,
    params: Any,
    fuel_mass_kg: float,
) -> NDArray[np.float64]:
    """Compute the thrust vector in the body frame.

    Action layout: ``[throttle, gimbal_pitch_cmd, gimbal_yaw_cmd]``.

    Gimbal angles deflect the thrust direction from the nominal +X_body
    (nose-forward) axis. With ε_p = gimbal_pitch_cmd · gimbal_max_rad and
    ε_y = gimbal_yaw_cmd · gimbal_max_rad, the body-frame thrust direction
    is

    .. code-block:: text

        F̂_thrust_body = ( cos(ε_p)·cos(ε_y),
                          cos(ε_p)·sin(ε_y),
                          −sin(ε_p) )

    Sign convention:
        - Positive ε_p tilts the thrust vector toward −Z_body, generating
          a positive pitch torque (nose up).
        - Positive ε_y tilts the thrust vector toward +Y_body, generating
          a positive yaw torque (nose right).

    Thrust magnitude is ``T = throttle · max_thrust_N`` with throttle
    clamped to ``[throttle_min, throttle_max]`` (and forced to zero when
    fuel is exhausted).

    Returns
    -------
    NDArray[(3,), float64]
        Thrust force in body-frame Newtons.
    """
    throttle = clamp_throttle(
        action[0], params.throttle_min, params.throttle_max, fuel_mass_kg
    )
    T = throttle * params.max_thrust_N
    eps_p = action[1] * params.gimbal_max_rad
    eps_y = action[2] * params.gimbal_max_rad
    cp, sp = np.cos(eps_p), np.sin(eps_p)
    cy, sy = np.cos(eps_y), np.sin(eps_y)
    return T * np.array([cp * cy, cp * sy, -sp])


def compute_gravity_inertial(total_mass_kg: float) -> NDArray[np.float64]:
    """Gravity force in inertial NED frame.

    NED Z+ points down, so gravity contributes ``(0, 0, +m·g)``. Constant
    g — see module-level ``G_EARTH``.
    """
    return np.array([0.0, 0.0, total_mass_kg * G_EARTH])


def compute_drag_inertial(
    velocity_inertial: NDArray[np.float64],
    params: Any,
    wind_velocity_ned: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Aerodynamic drag in the inertial frame.

    Simplified scalar model, computed on the **airspeed** — the vehicle's
    velocity relative to the surrounding air mass:

    .. code-block:: text

        v_air  = v − v_wind
        F_drag = −½ · ρ · ‖v_air‖ · v_air · C_d · A_ref

    A steady wind is modelled as a moving air mass (``wind_velocity_ned``, the
    air-mass velocity in NED). With ``wind_velocity_ned = None`` (or the zero
    vector) this reduces to drag on the inertial velocity — the nominal,
    disturbance-free case. This is the physically honest way to inject wind: the
    disturbance grid specifies wind in m/s precisely because it acts through the
    relative-airspeed drag term (see :mod:`robustness.disturbances`).

    Atmospheric density ρ is constant (sea-level ISA); no Mach-number,
    angle-of-attack, or altitude dependence — explicitly out of scope for
    moderate fidelity (see README §Limitations).

    Returns the zero vector when the airspeed is ~zero (avoids division by zero
    on the direction normalisation, which the algebra above implicitly does).
    """
    v_air = velocity_inertial if wind_velocity_ned is None else velocity_inertial - wind_velocity_ned
    airspeed = float(np.linalg.norm(v_air))
    if airspeed < 1e-9:
        return np.zeros(3)
    return (
        -0.5
        * RHO_AIR_SEA_LEVEL
        * airspeed
        * v_air
        * params.drag_coefficient
        * params.reference_area_m2
    )


def compute_gimbal_torque_body(
    thrust_force_body: NDArray[np.float64],
    engine_lever_arm_m: float,
) -> NDArray[np.float64]:
    """Body-frame torque produced by a gimballed thrust acting below the CoM.

    The engine acts at ``r = (−L, 0, 0)`` from the centre of mass, where
    L = ``engine_lever_arm_m``. Body-frame torque is ``τ = r × F``:

    .. code-block:: text

        r = (−L, 0, 0)
        F = (F_x, F_y, F_z)

        τ = (  0·F_z − 0·F_y,         # = 0
              0·F_x − (−L)·F_z,        # = +L·F_z
              (−L)·F_y − 0·F_x  )      # = −L·F_y

    So ``τ = (0, L·F_z, −L·F_y)``.

    Critical observation: ``τ_x = 0`` *always*. A single gimballed engine
    cannot generate a roll torque about the rocket's longitudinal axis.
    Real Falcon-9-class vehicles use RCS thrusters for roll control —
    explicitly out of scope for moderate fidelity. See README §Limitations.
    """
    return np.array(
        [
            0.0,
            engine_lever_arm_m * thrust_force_body[2],
            -engine_lever_arm_m * thrust_force_body[1],
        ]
    )


def compute_angular_acceleration(
    omega_body: NDArray[np.float64],
    torque_body: NDArray[np.float64],
    inertia_diag: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Euler's equations for a rigid body with principal-axes inertia.

    .. code-block:: text

        I · ω̇ + ω × (I · ω) = τ           (body frame)

    Component-wise with diagonal inertia ``I = diag(I_xx, I_yy, I_zz)``:

    .. code-block:: text

        ω̇_x = ( τ_x − (I_zz − I_yy) · ω_y · ω_z ) / I_xx
        ω̇_y = ( τ_y − (I_xx − I_zz) · ω_z · ω_x ) / I_yy
        ω̇_z = ( τ_z − (I_yy − I_xx) · ω_x · ω_y ) / I_zz

    The cross-coupling ``ω × (I·ω)`` is the body-frame manifestation of
    gyroscopic / inertial coupling — it is *not* a fictitious force, it's
    what Euler's equations say about a rotating reference frame.

    Reference: Greenwood, *Advanced Dynamics* §8.3 ("Euler's equations of
    motion for a rigid body").
    """
    Ixx, Iyy, Izz = inertia_diag
    p, q, r = omega_body
    tx, ty, tz = torque_body
    return np.array(
        [
            (tx - (Izz - Iyy) * q * r) / Ixx,
            (ty - (Ixx - Izz) * r * p) / Iyy,
            (tz - (Iyy - Ixx) * p * q) / Izz,
        ]
    )


# --- composed state derivative + integrator --------------------------------

def state_derivative(
    state: State,
    action: Action,
    params: Any,
    wind_velocity_ned: NDArray[np.float64] | None = None,
) -> State:
    """Time derivative of the 14-dim state vector.

    Composes translational, rotational, attitude, and mass-depletion
    derivatives. Pure function — no side effects.

    ``wind_velocity_ned`` (optional, m/s NED) is a steady air-mass velocity fed
    to the drag term as relative airspeed; ``None`` is the nominal no-wind case.

    The mass derivative uses the Tsiolkovsky surrogate

    .. code-block:: text

        ṁ_fuel = −T / (I_sp · g₀)

    where T is the *magnitude* of the body-frame thrust (which already
    accounts for throttle, fuel-empty cutoff, and gimbal angles — the
    gimbal angles barely affect magnitude because |cos(ε)| ≈ 1 for the
    small gimbal range, but we use the true magnitude for correctness).
    """
    v = velocity(state)
    q = quaternion(state)
    omega = angular_rate(state)
    m_fuel = fuel_mass(state)

    # Body-frame thrust + torque
    F_thrust_body = compute_thrust_force_body(action, params, m_fuel)
    torque_body = compute_gimbal_torque_body(F_thrust_body, params.engine_lever_arm_m)

    # Inertial-frame forces
    F_thrust_inertial = quat_rotate_body_to_inertial(q, F_thrust_body)
    total_mass = params.dry_mass_kg + m_fuel
    F_gravity = compute_gravity_inertial(total_mass)
    F_drag = compute_drag_inertial(v, params, wind_velocity_ned)
    F_total_inertial = F_thrust_inertial + F_gravity + F_drag

    # Translational derivatives
    pos_dot = v
    vel_dot = F_total_inertial / total_mass

    # Rotational derivatives
    quat_dot = quat_kinematics_dot(q, omega)
    inertia_diag = np.array(
        [params.inertia_xx, params.inertia_yy, params.inertia_zz]
    )
    omega_dot = compute_angular_acceleration(omega, torque_body, inertia_diag)

    # Mass depletion (Tsiolkovsky surrogate). Zero when no fuel or no thrust.
    thrust_magnitude = float(np.linalg.norm(F_thrust_body))
    if m_fuel > 0.0 and thrust_magnitude > 0.0:
        mass_dot = -thrust_magnitude / (params.isp_s * G0_TSIOLKOVSKY)
    else:
        mass_dot = 0.0

    return np.concatenate([pos_dot, vel_dot, quat_dot, omega_dot, [mass_dot]])


def rk4_step(
    state: State,
    action: Action,
    params: Any,
    dt: float,
    wind_velocity_ned: NDArray[np.float64] | None = None,
) -> State:
    """Advance the state by ``dt`` using classical 4th-order Runge-Kutta.

    Standard RK4 blend with zero-order-hold action across the step:

    .. code-block:: text

        k1 = f(y_n,           u)
        k2 = f(y_n + dt/2·k1, u)
        k3 = f(y_n + dt/2·k2, u)
        k4 = f(y_n + dt·k3,   u)
        y_{n+1} = y_n + (dt/6) · (k1 + 2·k2 + 2·k3 + k4)

    Local truncation error is O(dt⁵), global error O(dt⁴) — adequate at
    the project's 5 ms substep (200 Hz physics) for the dynamics ranges
    we exercise.

    ``wind_velocity_ned`` (optional, m/s NED) is held constant across the step
    alongside the action and forwarded to every stage's :func:`state_derivative`;
    ``None`` is the nominal no-wind case.

    Post-integration housekeeping:
        - Renormalise the quaternion to enforce ‖q‖ = 1 under floating-
          point drift.
        - Clamp fuel mass at zero (Tsiolkovsky can drive it negative if
          the command sustains thrust past empty).
    """
    k1 = state_derivative(state, action, params, wind_velocity_ned)
    k2 = state_derivative(state + (dt / 2.0) * k1, action, params, wind_velocity_ned)
    k3 = state_derivative(state + (dt / 2.0) * k2, action, params, wind_velocity_ned)
    k4 = state_derivative(state + dt * k3, action, params, wind_velocity_ned)

    state_next = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Renormalise quaternion (drift-correction).
    q = state_next[6:10]
    state_next[6:10] = q / np.linalg.norm(q)

    # Clamp fuel mass at zero (no negative-fuel).
    if state_next[13] < 0.0:
        state_next[13] = 0.0

    return state_next
