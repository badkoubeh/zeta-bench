"""Fixed-bound observation scaler.

Deterministic normalisation using known physical bounds. Each observation
dimension is divided by the corresponding bound from ``configs/env.yaml::obs_scaler``.
The same scaler is used by PID, SAC, PPO, and the adversary so behaviour
matches across controllers and between train/eval splits.

Layout matches :mod:`envs.rocket_landing_env`:

.. code-block:: text

    [0:2]    position_xy   /  position_xy_m       (m  → ~[-1, 1])
    [2]      position_z    /  position_z_m        (m  → ~[-1, 1])
    [3:6]    velocity_NED  /  velocity_mps        (m/s → ~[-1, 1])
    [6:9]    euler_attitude /  attitude_rad       (rad → ~[-1, 1])
    [9:12]   angular_rate  /  angular_rate_rad_s  (rad/s → ~[-1, 1])
    [12]     throttle_cmd  /  1.0                 (already in [0, 1])
    [13:15]  gimbal_cmds   /  1.0                 (already in [-1, 1])
    [15]     fuel_mass     /  fuel_mass_kg        (kg → ~[0, 1])
    [16]     fuel_remaining/  1.0                 (already in [0, 1])
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

OBS_DIM: int = 17


class FixedObsScaler:
    """Deterministic 17-dim observation scaler using known physical bounds."""

    def __init__(self, cfg: DictConfig) -> None:
        """Build the 17-element bounds vector from ``cfg.env.obs_scaler``."""
        c = cfg.env.obs_scaler
        self._bounds: NDArray[np.float64] = np.array(
            [
                c.position_xy_m,        # 0  x
                c.position_xy_m,        # 1  y
                c.position_z_m,         # 2  z
                c.velocity_mps,         # 3  vx
                c.velocity_mps,         # 4  vy
                c.velocity_mps,         # 5  vz
                c.attitude_rad,         # 6  roll
                c.attitude_rad,         # 7  pitch
                c.attitude_rad,         # 8  yaw
                c.angular_rate_rad_s,   # 9  omega_x
                c.angular_rate_rad_s,   # 10 omega_y
                c.angular_rate_rad_s,   # 11 omega_z
                1.0,                    # 12 throttle_cmd (passthrough)
                1.0,                    # 13 gimbal_pitch_cmd (passthrough)
                1.0,                    # 14 gimbal_yaw_cmd (passthrough)
                c.fuel_mass_kg,         # 15 fuel_mass (kg)
                1.0,                    # 16 fuel_remaining (passthrough)
            ],
            dtype=np.float64,
        )

    def scale(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalise a 17-dim observation by per-dim physical bounds.

        Returns a new array; does not mutate the input.
        """
        return obs / self._bounds

    def unscale(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Invert :meth:`scale` — multiplies by the per-dim bounds.

        Returns a new array; does not mutate the input.
        """
        return obs * self._bounds
