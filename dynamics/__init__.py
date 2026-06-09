"""Rocket dynamics package.

Self-contained 6-DOF rigid-body dynamics. Only ``envs/`` may import from this
package; nothing in this package may import from ``envs/``, ``controllers/``,
``adversary/``, or ``experiments/``.

Conventions
-----------
- Inertial frame: NED (X = North, Y = East, Z = Down). Gravity is +Z.
- Body frame: FRD (X = Forward, Y = Right, Z = Down).
- Attitude: unit quaternion (w-first) internally; Euler angles exposed in
  observations only.
- Integration: classical RK4 with ``physics_substeps`` substeps per control tick.
- Units: SI throughout (m, m/s, kg, rad, N, s).
"""
from __future__ import annotations

from dynamics.base import RocketDynamics
from dynamics.moderate_fidelity import ModerateFidelityDynamics, ModerateFidelityParams
from dynamics.types import ACTION_DIM, STATE_DIM, Action, DynamicsParams, State

__all__ = [
    "ACTION_DIM",
    "STATE_DIM",
    "Action",
    "DynamicsParams",
    "ModerateFidelityDynamics",
    "ModerateFidelityParams",
    "RocketDynamics",
    "State",
]
