"""Classical PID baseline for rocket landing.

Cascaded architecture (placeholder gains — to be tuned via step-response /
Ziegler-Nichols):

- **Altitude / descent-velocity loop** → throttle command
- **Lateral loop** → desired tilt → (chains into attitude loop, future)
- **Attitude loop** → gimbal commands

In this initial implementation only the **altitude / descent-velocity loop**
is active. Lateral and attitude loops are placeholders returning zero gimbal
commands — to be filled in during tuning. Even so, the controller
runs end-to-end and exposes the full PID structure (integrator state,
anti-windup clamping, derivative on error) for later extension.

Note: the placeholder gains will NOT land the rocket. The baseline tests
assert only that the loop runs without exception. PID tuning is a
follow-up task.

State (carried across `predict()` calls; cleared by `reset()`):
    - altitude integrator
    - previous descent-velocity error (for derivative term)

Anti-windup: integrator clamped to ±``integral_max``.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from utils.normalisation import FixedObsScaler


class PIDController:
    """Cascaded PID controller. Consumes the env's 17-dim scaled observation
    (the constructor takes a config with both ``env.obs_scaler`` bounds and
    ``pid_controller`` gains so the controller can unscale obs to physical
    units internally and then act).
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Construct from a Hydra config exposing ``pid_controller`` and
        ``env.episode.control_hz`` (for the derivative term's `dt`)."""
        self._cfg = cfg

        # Altitude / descent-velocity gains
        alt = cfg.pid_controller.altitude
        self._Kp_alt = float(alt.Kp)
        self._Ki_alt = float(alt.Ki)
        self._Kd_alt = float(alt.Kd)
        self._target_descent_mps = float(alt.target_descent_mps)

        # Anti-windup
        self._integral_max = float(cfg.pid_controller.integral_max)

        # dt for derivative (assumes fixed control rate)
        self._dt = 1.0 / float(cfg.env.episode.control_hz)

        # Need to unscale obs to physical units before acting
        self._scaler = FixedObsScaler(cfg)

        # Integrator state
        self._alt_integral: float = 0.0
        self._prev_vz_error: float = 0.0
        self._has_prev_error: bool = False

    def reset(self) -> None:
        """Clear integrator and derivative state at episode start."""
        self._alt_integral = 0.0
        self._prev_vz_error = 0.0
        self._has_prev_error = False

    def predict(
        self,
        obs: NDArray[np.float64],
        deterministic: bool = True,
    ) -> NDArray[np.float64]:
        """Compute a 3-dim action from a 17-dim scaled observation.

        Currently only the altitude / descent-velocity loop is active.
        Lateral and attitude commands are placeholders (zero gimbal). The
        ``deterministic`` flag is accepted for interface parity with the
        SB3 agents; PID has no stochastic component, so it is ignored.
        """
        # Unscale to physical units
        raw = self._scaler.unscale(obs)
        vz = float(raw[5])  # NED z velocity, positive = descending

        # Altitude / descent-velocity PID
        vz_error = vz - self._target_descent_mps
        self._alt_integral += vz_error * self._dt
        self._alt_integral = float(
            np.clip(self._alt_integral, -self._integral_max, self._integral_max)
        )
        if self._has_prev_error:
            vz_error_dot = (vz_error - self._prev_vz_error) / self._dt
        else:
            vz_error_dot = 0.0
        self._prev_vz_error = vz_error
        self._has_prev_error = True

        throttle_cmd = (
            self._Kp_alt * vz_error
            + self._Ki_alt * self._alt_integral
            + self._Kd_alt * vz_error_dot
        )
        # Bound to env's action space [0, 1] for throttle. The dynamics
        # layer further clamps to [throttle_min, throttle_max] or maps
        # sub-threshold commands to engine-off (zero thrust).
        throttle_cmd = float(np.clip(throttle_cmd, 0.0, 1.0))

        # Lateral + attitude loops: not active yet — wired in during
        # tuning. For now, no gimbal command.
        gimbal_pitch_cmd = 0.0
        gimbal_yaw_cmd = 0.0

        return np.array([throttle_cmd, gimbal_pitch_cmd, gimbal_yaw_cmd], dtype=np.float64)

    def save(self, path: str) -> None:
        """Persist gains to disk (YAML)."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "PIDController":
        """Restore controller from saved gains."""
        raise NotImplementedError
