"""Moderate-fidelity 6-DOF rigid-body rocket dynamics.

Models
------
- Translational: Newton's second law in the NED inertial frame. Body-frame
  thrust rotated into inertial via the attitude quaternion; constant
  gravity (+Z in NED); scalar aerodynamic drag with constant sea-level
  density.
- Rotational: Euler's equations in the body frame; gimballed thrust
  produces body-frame torques about the centre of mass.
- Fuel depletion: ``ṁ = −T / (I_sp · g₀)`` (Tsiolkovsky surrogate). The
  inertia tensor is NOT updated as fuel burns — explicitly a moderate-
  fidelity simplification (see ``README.md`` §Limitations).
- Integration: classical RK4 with ``physics_substeps`` substeps per
  control tick. Quaternion renormalised after each substep.

The actual mathematics lives in :mod:`dynamics.equations_of_motion`; this
module assembles a parameter container and binds the abstract interface.

Frames + column layouts: see :mod:`dynamics.types`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from dynamics.base import RocketDynamics
from dynamics.equations_of_motion import rk4_step
from dynamics.types import Action, DynamicsParams, State


@dataclass(frozen=True, slots=True)
class ModerateFidelityParams:
    """Immutable parameter set for :class:`ModerateFidelityDynamics`.

    All values SI. See ``docs/parameter_sources.md`` for provenance of
    each value (Merlin-1D thrust, Falcon-9 mass distribution, ISA air
    density, etc.).
    """

    dry_mass_kg: float
    initial_fuel_kg: float
    max_thrust_N: float
    isp_s: float
    drag_coefficient: float
    reference_area_m2: float
    inertia_xx: float
    inertia_yy: float
    inertia_zz: float
    gimbal_max_rad: float
    throttle_min: float
    throttle_max: float
    physics_substeps: int
    engine_lever_arm_m: float


class ModerateFidelityDynamics(RocketDynamics):
    """6-DOF rigid-body dynamics, moderate-fidelity tier.

    Composes the pure functions from :mod:`dynamics.equations_of_motion`
    into a stateful object that owns its parameter set and exposes the
    :class:`RocketDynamics` interface.

    The class itself holds NO mutable state across calls — every call to
    :meth:`step` is functionally pure given ``(state, action, dt)``. The
    object exists only to bundle the parameters.
    """

    def __init__(self, params: ModerateFidelityParams) -> None:
        """Construct dynamics from an immutable parameter dataclass."""
        self._params = params

    def step(self, state: State, action: Action, dt: float) -> State:
        """Integrate one control tick using RK4 with ``physics_substeps`` substeps.

        The control-tick ``dt`` is subdivided into ``physics_substeps`` equal
        substeps. Action is held constant across substeps (zero-order hold).
        After each substep the integrator renormalises the quaternion and
        clamps fuel mass at zero (see :func:`rk4_step`).

        Parameters
        ----------
        state : State
            Current 14-dim state. See :mod:`dynamics.types` for layout.
        action : Action
            3-dim action ``[throttle, gimbal_pitch_cmd, gimbal_yaw_cmd]``,
            held constant across substeps.
        dt : float
            Control-tick duration in seconds (e.g. 0.02 for 50 Hz).

        Returns
        -------
        State
            Next 14-dim state.
        """
        n_sub = int(self._params.physics_substeps)
        dt_sub = dt / n_sub
        s = state.copy()
        for _ in range(n_sub):
            s = rk4_step(s, action, self._params, dt_sub)
        return s

    def get_params(self) -> DynamicsParams:
        """Return parameters as a flat float64 ndarray.

        Order matches the field order of :class:`ModerateFidelityParams`.
        Used by the env wrapper for logging and by adversary policies that
        condition on vehicle parameters (a future extension).
        """
        values = list(asdict(self._params).values())
        return np.asarray(values, dtype=np.float64)
