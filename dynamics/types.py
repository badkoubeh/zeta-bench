"""Type aliases, column layouts, and pure accessors for the 14-dim state vector.

Public dynamics APIs operate on plain ``numpy.ndarray`` (float64) buffers for
performance and SB3/Gymnasium compatibility. Layouts are documented here and
enforced by convention; every implementation of :class:`RocketDynamics` must
respect them.

State layout (length 14)
------------------------
    [0:3]    position_NED          (m)        — North, East, Down
    [3:6]    velocity_NED          (m/s)
    [6:10]   attitude_quat         (—)        — unit quaternion, Hamilton (w, x, y, z)
    [10:13]  angular_rate_body     (rad/s)    — roll, pitch, yaw rates in body frame
    [13]     fuel_mass             (kg)

Action layout (length 3)
------------------------
    [0]   throttle              ∈ [0, 1]
    [1]   gimbal_pitch_cmd      ∈ [-1, 1]   — scaled to ±gimbal_max_rad
    [2]   gimbal_yaw_cmd        ∈ [-1, 1]   — scaled to ±gimbal_max_rad

Frame conventions
-----------------
- Inertial: NED — X = North, Y = East, Z = Down. Gravity g = +9.81 ẑ.
- Body: FRD — X = Forward (rocket nose), Y = Right, Z = Down (toward belly).
- Body-to-inertial rotation via the attitude quaternion (Hamilton convention).

The "nose-up" attitude (body +X aligned with inertial −Z) is *not* the identity
quaternion — it is a +90° rotation about the inertial Y axis, i.e.
q = (cos 45°, 0, sin 45°, 0) = (√2/2, 0, √2/2, 0).

Observation layout (length 17) is built by :mod:`envs.rocket_landing_env`,
not this package — see that module's docstring for the obs columns.
"""
from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

State: TypeAlias = NDArray[np.float64]
Action: TypeAlias = NDArray[np.float64]
DynamicsParams: TypeAlias = NDArray[np.float64]

# Dimensions
STATE_DIM: int = 14
ACTION_DIM: int = 3

# Column-index slices into the state vector (used by accessors and by
# integrators that need to update specific blocks in-place).
POS_SLICE: slice = slice(0, 3)
VEL_SLICE: slice = slice(3, 6)
QUAT_SLICE: slice = slice(6, 10)
OMEGA_SLICE: slice = slice(10, 13)
FUEL_IDX: int = 13


# --- pure accessors ---------------------------------------------------------
# These return slices into the underlying buffer; mutate only via assignment
# to the slice (state[POS_SLICE] = ...) to avoid aliasing surprises.

def position(state: State) -> NDArray[np.float64]:
    """Return position in NED inertial frame (m), shape ``(3,)``."""
    return state[POS_SLICE]


def velocity(state: State) -> NDArray[np.float64]:
    """Return velocity in NED inertial frame (m/s), shape ``(3,)``."""
    return state[VEL_SLICE]


def quaternion(state: State) -> NDArray[np.float64]:
    """Return body-to-inertial attitude quaternion ``(w, x, y, z)``, shape ``(4,)``."""
    return state[QUAT_SLICE]


def angular_rate(state: State) -> NDArray[np.float64]:
    """Return body-frame angular rate (rad/s), shape ``(3,)``."""
    return state[OMEGA_SLICE]


def fuel_mass(state: State) -> float:
    """Return current fuel mass (kg) as a scalar."""
    return float(state[FUEL_IDX])


# --- constructor helper -----------------------------------------------------

def make_state(
    position_NED: NDArray[np.float64],
    velocity_NED: NDArray[np.float64],
    quat_wxyz: NDArray[np.float64],
    angular_rate_body: NDArray[np.float64],
    fuel_mass_kg: float,
) -> State:
    """Assemble a 14-dim state vector from named components.

    Useful for test fixtures and for any caller that wants to construct a
    state without remembering the index layout. The returned array is
    float64 and contiguous.
    """
    return np.concatenate(
        [
            np.asarray(position_NED, dtype=np.float64),
            np.asarray(velocity_NED, dtype=np.float64),
            np.asarray(quat_wxyz, dtype=np.float64),
            np.asarray(angular_rate_body, dtype=np.float64),
            np.array([fuel_mass_kg], dtype=np.float64),
        ]
    )


# --- well-known attitudes ---------------------------------------------------

# Quaternion for "rocket nose pointed straight up" (body +X aligned with
# inertial −Z). +90° rotation about the inertial Y axis.
UPRIGHT_QUAT: NDArray[np.float64] = np.array(
    [np.sqrt(2) / 2, 0.0, np.sqrt(2) / 2, 0.0], dtype=np.float64
)

# Identity quaternion (body and inertial frames coincide).
IDENTITY_QUAT: NDArray[np.float64] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
