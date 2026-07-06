"""Training-time domain randomisation wrapper.

Draws a fresh disturbance for every training episode and pushes it into the
wrapped :class:`~envs.rocket_landing_env.RocketLandingEnv` via its primitive
``set_disturbance`` hook, so a model-free agent learns *across* the disturbance
distribution rather than only on nominal dynamics. This is the standard fix for
the train/test distribution shift that makes a nominally-trained policy fragile
to unseen wind / mass / sensor-noise conditions.

Scope + layering
----------------
- **Training only.** The agent wrappers apply this to the *training* vec-env when
  ``cfg.env.domain_randomization.enabled`` is true. The eval / model-selection env
  and the graduated robustness matrix build the bare env, so evaluation stays
  deterministic and comparable (disturbance is the sole controlled variable there).
- The wrapper samples primitive values and calls ``set_disturbance`` — it does not
  import :mod:`robustness`, mirroring the env's own polar-wind convention
  (0°=N=+X, 90°=E=+Y) so the ``envs`` layer stays self-contained.

Ranges come from ``cfg.env.domain_randomization`` (see ``configs/env.yaml``); the
extreme sensor-noise regime (σ≥0.10) is intentionally excluded upstream because it
is a shared physics/observability wall no controller survives.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf


def _pair(cfg: DictConfig, key: str, default: tuple[float, float]) -> tuple[float, float]:
    """Read a ``[low, high]`` range from config, falling back to ``default``."""
    val = OmegaConf.select(cfg, key, default=None)
    if val is None:
        return default
    seq = list(val)
    return float(seq[0]), float(seq[1])


class DomainRandomizationWrapper(gym.Wrapper):
    """Resample a disturbance from configured ranges on every ``reset``.

    Parameters
    ----------
    env : gym.Env
        A :class:`RocketLandingEnv` (or wrapper thereof) exposing
        ``set_disturbance``.
    dr_cfg : DictConfig
        The ``env.domain_randomization`` config block with the sampling ranges.
    """

    def __init__(self, env: gym.Env, dr_cfg: DictConfig) -> None:
        super().__init__(env)
        self._wind_mag = _pair(dr_cfg, "wind_magnitude_mps", (0.0, 10.0))
        self._mass = _pair(dr_cfg, "mass_offset_fraction", (-0.20, 0.20))
        self._sigma = _pair(dr_cfg, "sensor_noise_sigma", (0.0, 0.05))
        self._spike_p = _pair(dr_cfg, "sensor_spike_probability", (0.0, 0.05))
        self._spike_mag = float(OmegaConf.select(dr_cfg, "sensor_spike_magnitude", default=0.5))
        delay = _pair(dr_cfg, "actuator_delay_steps", (0.0, 0.0))
        self._delay = (int(delay[0]), int(delay[1]))
        # Disturbance-severity curriculum: scale the ranges by a fraction that ramps
        # 0→1 over this many PER-ENV steps (0 disables the ramp → full ranges always).
        self._severity_anneal = int(
            OmegaConf.select(dr_cfg, "severity_anneal_steps", default=0)
        )
        self._steps = 0
        # Dedicated RNG so disturbance sampling is independent of the env's own
        # IC/sensor-noise stream; seeded from the reset seed when one is supplied.
        self._dr_rng = np.random.default_rng()

    def _severity(self) -> float:
        """Current severity fraction in [0, 1] (1.0 when the ramp is disabled)."""
        if self._severity_anneal <= 0:
            return 1.0
        return float(min(1.0, self._steps / self._severity_anneal))

    def _sample_wind_ned(self, severity: float) -> NDArray[np.float64] | None:
        """Sample a horizontal NED wind velocity (or ``None`` when magnitude is 0)."""
        magnitude = float(self._dr_rng.uniform(self._wind_mag[0], self._wind_mag[1] * severity))
        if magnitude == 0.0:
            return None
        bearing = float(self._dr_rng.uniform(0.0, 360.0))
        theta = np.deg2rad(bearing)
        return np.array(
            [magnitude * np.cos(theta), magnitude * np.sin(theta), 0.0],
            dtype=np.float64,
        )

    def step(self, action):
        """Delegate to the wrapped env, counting steps for the severity ramp."""
        self._steps += 1
        return self.env.step(action)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Draw a fresh (severity-scaled) disturbance, apply it, then reset the env."""
        if seed is not None:
            self._dr_rng = np.random.default_rng(seed)

        severity = self._severity()
        delay_lo, delay_hi = self._delay
        delay_hi = int(round(delay_hi * severity))
        actuator_delay = (
            int(self._dr_rng.integers(delay_lo, delay_hi + 1)) if delay_hi > delay_lo else delay_lo
        )
        self.env.set_disturbance(
            wind_velocity_ned=self._sample_wind_ned(severity),
            mass_offset_fraction=float(
                self._dr_rng.uniform(self._mass[0] * severity, self._mass[1] * severity)
            ),
            sensor_noise_sigma=float(
                self._dr_rng.uniform(self._sigma[0], self._sigma[1] * severity)
            ),
            sensor_spike_probability=float(
                self._dr_rng.uniform(self._spike_p[0], self._spike_p[1] * severity)
            ),
            sensor_spike_magnitude=self._spike_mag,
            actuator_delay_steps=actuator_delay,
        )
        return self.env.reset(seed=seed, options=options)


def wrap_if_enabled(env: gym.Env, cfg: DictConfig) -> gym.Env:
    """Wrap ``env`` in :class:`DomainRandomizationWrapper` iff enabled in config.

    Returns the env unchanged when ``env.domain_randomization`` is absent or
    ``enabled`` is false, so nominal training and all evaluation paths behave
    exactly as before.
    """
    dr_cfg = OmegaConf.select(cfg, "env.domain_randomization", default=None)
    if dr_cfg is not None and bool(OmegaConf.select(dr_cfg, "enabled", default=False)):
        return DomainRandomizationWrapper(env, dr_cfg)
    return env
