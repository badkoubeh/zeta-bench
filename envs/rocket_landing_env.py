"""Rocket landing Gymnasium environment.

Observation space (17-dim ``Box``, all values normalised by FixedObsScaler):

.. code-block:: text

    [0:3]    position_NED          (m / position bounds)
    [3:6]    velocity_NED          (m/s / velocity_mps)
    [6:9]    euler_attitude        (rad / attitude_rad)
    [9:12]   angular_rate_body     (rad/s / angular_rate_rad_s)
    [12:15]  last_action           [throttle ∈ [0,1], gimbal_pitch ∈ [-1,1], gimbal_yaw ∈ [-1,1]]
    [15]     fuel_mass             (kg / fuel_mass_kg)
    [16]     fuel_remaining        ∈ [0, 1] (passthrough)

Action space (3-dim ``Box``):

.. code-block:: text

    [0]  throttle               ∈ [0, 1]
    [1]  gimbal_pitch_command   ∈ [-1, 1]  (scaled to ±gimbal_max_rad in dynamics)
    [2]  gimbal_yaw_command     ∈ [-1, 1]

Reward: potential-based shaping (per step) + impact-aware terminal outcome.
See :mod:`envs.reward`.

Disturbances: the env is nominal (disturbance-free) by default. Four typed
disturbances can be injected — wind (relative-airspeed drag), mass uncertainty
(dry-mass offset), sensor noise (Gaussian + spikes on the observation), and
actuator delay (latency on the applied action). They are supplied through the
primitive :meth:`RocketLandingEnv.set_disturbance` hook or the optional
``cfg.env.disturbance`` block; the graduated robustness matrix drives them
per-cell. To respect the layering (``envs`` sits below ``robustness``), the env
takes only primitive values here — the typed ``Disturbance`` object and the
grid live in :mod:`robustness.disturbances`.

Termination:
    - ``success`` — z >= 0 (touchdown at/below pad) with low velocity, tilt, ω
    - ``crash``   — z >= 0 but landing thresholds violated
    - ``out_of_bounds`` — lateral radius > cylinder OR z < −ceiling

Truncation:
    - ``timeout`` — step count reached ``cfg.env.episode.max_steps``

Import rule: this module imports from ``dynamics/`` and ``utils/`` only.
"""
from __future__ import annotations

import dataclasses
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig, OmegaConf

from dynamics.equations_of_motion import quat_to_euler
from dynamics.moderate_fidelity import ModerateFidelityDynamics, ModerateFidelityParams
from dynamics.types import (
    ACTION_DIM,
    Action,
    State,
    angular_rate,
    fuel_mass,
    make_state,
    position,
    quaternion,
    velocity,
)
from envs.curriculum import Curriculum
from envs.reward import (
    shaping_reward,
    terminal_reward,
    tilt_from_vertical,
    touchdown_metrics,
)
from utils.normalisation import FixedObsScaler

OBS_DIM: int = 17

_UP_INERTIAL = np.array([0.0, 0.0, -1.0], dtype=np.float64)
_BODY_NOSE = np.array([1.0, 0.0, 0.0], dtype=np.float64)


class RocketLandingEnv(gym.Env):
    """Gymnasium env wrapping :class:`ModerateFidelityDynamics`."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(self, cfg: DictConfig) -> None:
        """Construct env from a composed Hydra config (must include
        ``env``, ``reward`` sections)."""
        super().__init__()
        self._cfg = cfg

        # Dynamics
        d = cfg.env.dynamics
        params = ModerateFidelityParams(
            dry_mass_kg=d.dry_mass_kg,
            initial_fuel_kg=d.initial_fuel_kg,
            max_thrust_N=d.max_thrust_N,
            isp_s=d.isp_s,
            drag_coefficient=d.drag_coefficient,
            reference_area_m2=d.reference_area_m2,
            inertia_xx=d.inertia_xx,
            inertia_yy=d.inertia_yy,
            inertia_zz=d.inertia_zz,
            gimbal_max_rad=d.gimbal_max_rad,
            throttle_min=d.throttle_min,
            throttle_max=d.throttle_max,
            physics_substeps=int(cfg.env.episode.physics_substeps),
            engine_lever_arm_m=d.engine_lever_arm_m,
        )
        # Base (nominal) parameters. Mass-uncertainty disturbances rebuild the
        # dynamics from these; keep them immutable so nominal is always recoverable.
        self._base_params = params
        self._dynamics = ModerateFidelityDynamics(params)
        self._scaler = FixedObsScaler(cfg)
        self._curriculum = Curriculum(cfg)

        # Disturbance state (nominal until set via set_disturbance / config).
        self._wind_velocity_ned: np.ndarray | None = None
        self._mass_offset_fraction: float = 0.0
        self._sensor_noise_sigma: float = 0.0
        self._sensor_spike_probability: float = 0.0
        self._sensor_spike_magnitude: float = 0.0
        self._actuator_delay_steps: int = 0
        self._action_buffer: deque[np.ndarray] | None = None

        # Episode bookkeeping
        self._max_steps: int = int(cfg.env.episode.max_steps)
        self._dt: float = 1.0 / float(cfg.env.episode.control_hz)
        self._global_step: int = 0  # accumulates across episodes; drives curriculum
        self._step_in_episode: int = 0
        self._state: State | None = None
        self._prev_action: Action | None = None
        self._initial_fuel: float = float(d.initial_fuel_kg)
        self._task_difficulty: float = 0.0
        # Discount used for potential-based reward shaping. Must match the
        # agent's gamma for PBRS policy-invariance; falls back to 0.99 when the
        # env is built without an agent section (e.g. some eval contexts).
        self._gamma: float = float(
            cfg.agent.gamma if "agent" in cfg and "gamma" in cfg.agent else 0.99
        )

        # Spaces
        # float32 observations: MPS has no float64 support, and float32 is the
        # SB3/Gymnasium convention. Internal physics stays float64 (see _build_obs).
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0], dtype=np.float64),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float64),
            dtype=np.float64,
        )

        self._rng = np.random.default_rng()

        # Apply the optional nominal disturbance block from config (if present).
        # The robustness matrix overrides this per-cell at runtime.
        dist_cfg = OmegaConf.select(cfg, "env.disturbance", default=None)
        if dist_cfg is not None:
            self._apply_disturbance_config(dist_cfg)

    # --- Disturbance control (primitive hook for robustness/) ---------------

    def set_disturbance(
        self,
        *,
        wind_velocity_ned: np.ndarray | None = None,
        mass_offset_fraction: float = 0.0,
        sensor_noise_sigma: float = 0.0,
        sensor_spike_probability: float = 0.0,
        sensor_spike_magnitude: float = 0.0,
        actuator_delay_steps: int = 0,
    ) -> None:
        """Set the active disturbance from primitive values (nominal defaults).

        This is the layering-safe hook the graduated matrix uses to switch cells
        cheaply: ``robustness.disturbances.Disturbance`` is unpacked into these
        primitives by the matrix runner (the env never imports ``robustness``).
        Takes effect from the next :meth:`reset`; the mass offset rebuilds the
        dynamics immediately. Omitting all arguments restores nominal.

        Parameters
        ----------
        wind_velocity_ned : ndarray or None
            Air-mass velocity (m/s, NED) for the relative-airspeed drag term.
        mass_offset_fraction : float
            Signed fraction offsetting the vehicle dry mass.
        sensor_noise_sigma, sensor_spike_probability, sensor_spike_magnitude : float
            Gaussian σ, spike probability, and spike size on the scaled obs.
        actuator_delay_steps : int
            Control-tick latency applied to the commanded action.
        """
        self._wind_velocity_ned = (
            None if wind_velocity_ned is None
            else np.asarray(wind_velocity_ned, dtype=np.float64)
        )
        self._sensor_noise_sigma = float(sensor_noise_sigma)
        self._sensor_spike_probability = float(sensor_spike_probability)
        self._sensor_spike_magnitude = float(sensor_spike_magnitude)
        self._actuator_delay_steps = int(actuator_delay_steps)
        self._apply_mass_offset(float(mass_offset_fraction))

    def _apply_mass_offset(self, fraction: float) -> None:
        """Rebuild the dynamics with dry mass scaled by ``1 + fraction``."""
        self._mass_offset_fraction = fraction
        scaled = dataclasses.replace(
            self._base_params,
            dry_mass_kg=self._base_params.dry_mass_kg * (1.0 + fraction),
        )
        self._dynamics = ModerateFidelityDynamics(scaled)

    def _apply_disturbance_config(self, dist_cfg: DictConfig) -> None:
        """Translate a ``cfg.env.disturbance`` block into a ``set_disturbance`` call.

        Wind is given in polar form (magnitude + compass bearing) and converted
        to a NED velocity here — mirroring ``robustness.disturbances.wind_from_polar``
        (0°=N=+X, 90°=E=+Y) without importing the higher robustness layer.
        """
        magnitude = float(OmegaConf.select(dist_cfg, "wind_magnitude_mps", default=0.0))
        bearing = float(OmegaConf.select(dist_cfg, "wind_direction_deg", default=0.0))
        if magnitude != 0.0:
            theta = np.deg2rad(bearing)
            wind = np.array([magnitude * np.cos(theta), magnitude * np.sin(theta), 0.0])
        else:
            wind = None
        self.set_disturbance(
            wind_velocity_ned=wind,
            mass_offset_fraction=float(
                OmegaConf.select(dist_cfg, "mass_offset_fraction", default=0.0)
            ),
            sensor_noise_sigma=float(
                OmegaConf.select(dist_cfg, "sensor_noise_sigma", default=0.0)
            ),
            sensor_spike_probability=float(
                OmegaConf.select(dist_cfg, "sensor_spike_probability", default=0.0)
            ),
            sensor_spike_magnitude=float(
                OmegaConf.select(dist_cfg, "sensor_spike_magnitude", default=0.0)
            ),
            actuator_delay_steps=int(
                OmegaConf.select(dist_cfg, "actuator_delay_steps", default=0)
            ),
        )

    # --- Gymnasium API -----------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Sample initial conditions from the curriculum-lerped envelope."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._task_difficulty = self._curriculum.task_difficulty(self._global_step)
        pos_NED, vel_NED, quat_init, omega_init = (
            self._curriculum.sample_initial_conditions(self._rng, self._task_difficulty)
        )

        self._state = make_state(
            position_NED=pos_NED,
            velocity_NED=vel_NED,
            quat_wxyz=quat_init,
            angular_rate_body=omega_init,
            fuel_mass_kg=self._initial_fuel,
        )
        self._step_in_episode = 0
        self._prev_action = None
        # Cold-start the actuator-delay line with neutral commands (only when a
        # delay is active). Applied action at step t is the command from
        # ``actuator_delay_steps`` ticks earlier.
        self._action_buffer = (
            deque(
                (np.zeros(ACTION_DIM, dtype=np.float64) for _ in range(self._actuator_delay_steps)),
                maxlen=self._actuator_delay_steps,
            )
            if self._actuator_delay_steps > 0
            else None
        )

        obs = self._build_obs(self._state, action=np.zeros(ACTION_DIM, dtype=np.float64))
        info: dict[str, Any] = {
            "task_difficulty": self._task_difficulty,
            "termination_reason": "reset",
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one control tick."""
        if self._state is None:
            raise RuntimeError("step() called before reset()")

        command = np.clip(action.astype(np.float64), self.action_space.low, self.action_space.high)
        # Actuator delay: the vehicle responds to a lagged command. Everything
        # downstream (dynamics, reward, obs last_action) uses the *applied*
        # action — the true actuator state. With delay 0 this is the command.
        applied = self._apply_actuator_delay(command)

        prev_state = self._state
        prev_fuel = fuel_mass(prev_state)
        next_state = self._dynamics.step(
            prev_state, applied, self._dt, self._wind_velocity_ned
        )
        self._state = next_state
        self._step_in_episode += 1
        self._global_step += 1

        terminated, truncated, reason = self._check_termination()

        # Potential-based shaping: on terminal transitions the next-state
        # potential is taken as 0 (absorbing-state convention) so the episode's
        # shaping telescopes to -Phi(s_0) and the outcome is carried entirely by
        # the terminal reward — this is what removes the old "crash early to stop
        # accumulating penalties" incentive.
        shaping_next = None if (terminated or truncated) else next_state
        dense, components = shaping_reward(
            prev_state,
            shaping_next,
            applied,
            self._prev_action,
            prev_fuel,
            self._gamma,
            self._cfg,
        )
        terminal = (
            terminal_reward(reason, next_state, self._cfg)
            if (terminated or truncated)
            else 0.0
        )
        components["terminal"] = terminal
        reward = float(dense + terminal)

        obs = self._build_obs(next_state, applied)
        info: dict[str, Any] = {
            "reward_components": components,
            "terminal_metrics": touchdown_metrics(next_state),
            "step_in_episode": self._step_in_episode,
            "global_step": self._global_step,
            "termination_reason": reason,
            "task_difficulty": self._task_difficulty,
        }

        self._prev_action = applied
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render the current frame.

        Deferred — the matplotlib MP4/GIF pipeline lands in a separate
        :mod:`utils.render` task. Returning None here keeps the env
        compatible with the Gymnasium render API without erroring.
        """
        return None

    # --- helpers -----------------------------------------------------------

    def _apply_actuator_delay(self, command: np.ndarray) -> np.ndarray:
        """Return the action actually applied this tick given the delay line.

        With no delay, the command is applied as-is. Otherwise the applied
        action is the command issued ``actuator_delay_steps`` ticks earlier; the
        buffer is cold-started with neutral commands in :meth:`reset`.
        """
        if self._action_buffer is None:
            return command
        applied = self._action_buffer[0]  # oldest queued command
        self._action_buffer.append(command)  # maxlen drops the one just applied
        return applied

    def _apply_sensor_noise(self, obs: np.ndarray) -> np.ndarray:
        """Add Gaussian noise and sparse spikes to the scaled observation.

        Noise is drawn from the env's seeded ``self._rng``, so a fixed seed
        reproduces the exact perturbation sequence — the fairness property the
        graduated matrix relies on. σ and spike size are in scaled-obs units
        (fractions of each sensor's full-scale range).
        """
        noisy = obs.astype(np.float32)
        if self._sensor_noise_sigma > 0.0:
            gaussian = self._rng.normal(0.0, self._sensor_noise_sigma, size=noisy.shape)
            noisy = noisy + gaussian.astype(np.float32)
        if self._sensor_spike_probability > 0.0:
            fired = self._rng.random(noisy.shape) < self._sensor_spike_probability
            signs = self._rng.choice(np.array([-1.0, 1.0]), size=noisy.shape)
            spikes = fired * signs * self._sensor_spike_magnitude
            noisy = noisy + spikes.astype(np.float32)
        return noisy

    def _build_obs(self, state: State, action: Action) -> np.ndarray:
        """Pack state + last action into the 17-dim scaled observation."""
        pos = position(state)
        vel = velocity(state)
        euler = quat_to_euler(quaternion(state))
        omega = angular_rate(state)
        fm = fuel_mass(state)
        fuel_remaining = fm / self._initial_fuel if self._initial_fuel > 0 else 0.0

        raw = np.concatenate(
            [
                pos,
                vel,
                euler,
                omega,
                action,
                np.array([fm, fuel_remaining], dtype=np.float64),
            ]
        )
        scaled = self._scaler.scale(raw).astype(np.float32)
        if self._sensor_noise_sigma > 0.0 or self._sensor_spike_probability > 0.0:
            scaled = self._apply_sensor_noise(scaled)
        return scaled

    def _check_termination(self) -> tuple[bool, bool, str]:
        """Return (terminated, truncated, reason)."""
        assert self._state is not None
        pos = position(self._state)

        # Out-of-bounds: cylinder radius or ceiling
        lateral = float(np.linalg.norm(pos[:2]))
        if lateral > float(self._cfg.env.oob.cylinder_radius_m):
            return True, False, "out_of_bounds"
        if pos[2] < -float(self._cfg.env.oob.ceiling_m):
            return True, False, "out_of_bounds"

        # Touchdown (z ≥ 0 in NED = at or below pad surface)
        if pos[2] >= 0.0:
            speed = float(np.linalg.norm(velocity(self._state)))
            omega_mag = float(np.linalg.norm(angular_rate(self._state)))
            tilt = tilt_from_vertical(self._state)
            td = self._cfg.env.touchdown
            if (
                speed <= float(td.velocity_threshold_mps)
                and tilt <= float(td.tilt_threshold_rad)
                and omega_mag <= float(td.angular_rate_threshold_rad_s)
            ):
                return True, False, "success"
            return True, False, "crash"

        # Timeout (truncation, not termination)
        if self._step_in_episode >= self._max_steps:
            return False, True, "timeout"

        return False, False, "ongoing"
