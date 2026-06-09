"""Abstract base class for rocket dynamics.

Defines the contract every dynamics implementation must satisfy:

- :meth:`RocketDynamics.step` — integrate one control tick forward.
- :meth:`RocketDynamics.get_params` — expose the immutable parameter vector.

Concrete subclasses (e.g. :class:`ModerateFidelityDynamics`, future
``HighFidelityDynamics``) implement the equations of motion at different
fidelity tiers; the env wrapper selects one via Hydra config
(``dynamics.fidelity``). Adding a new tier must not require changes outside
``dynamics/``.

Import rule: only ``envs/`` may import from this module.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from dynamics.types import Action, DynamicsParams, State


class RocketDynamics(ABC):
    """Abstract 6-DOF rocket dynamics model."""

    @abstractmethod
    def step(self, state: State, action: Action, dt: float) -> State:
        """Integrate dynamics forward by one control tick.

        Parameters
        ----------
        state : State
            Current 14-dim state vector. See :mod:`dynamics.types` for layout.
        action : Action
            3-dim action vector ``[throttle, gimbal_pitch, gimbal_yaw]``.
        dt : float
            Control-tick duration in seconds (e.g. ``0.02`` for 50 Hz).

        Returns
        -------
        State
            Next 14-dim state vector.
        """
        raise NotImplementedError

    @abstractmethod
    def get_params(self) -> DynamicsParams:
        """Return the immutable vehicle/environment parameters as a flat array.

        Used by the env wrapper for logging and by adversary policies that
        condition on vehicle parameters.
        """
        raise NotImplementedError
