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

Ranges come from ``cfg.env.domain_randomization`` (see ``configs/env.yaml``) and are
**required** when DR is enabled — the wrapper raises on a missing range rather than
silently substituting a hardcoded default, so config stays the single source of
disturbance magnitudes (set a channel to ``[0.0, 0.0]`` to leave it nominal). The
extreme sensor-noise regime (σ≥0.10) is intentionally excluded upstream because it
is a shared physics/observability wall no controller survives.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf


def _require_pair(cfg: DictConfig, key: str) -> tuple[float, float]:
    """Read a required ``[low, high]`` range from the domain-randomisation config.

    Raises :class:`ValueError` if the key is absent. ``configs/env.yaml`` is the
    single source of disturbance magnitudes: rather than silently substituting a
    hardcoded default (which could diverge from the config and train the policy on
    an unintended disturbance), a missing range is treated as a configuration
    error. To leave a channel nominal, set it explicitly to ``[0.0, 0.0]``.
    """
    val = OmegaConf.select(cfg, key, default=None)
    if val is None:
        raise ValueError(
            f"env.domain_randomization.{key} is required when domain randomisation is "
            f"enabled; set it explicitly ([0.0, 0.0] leaves that channel nominal)."
        )
    seq = list(val)
    return float(seq[0]), float(seq[1])


def _require_scalar(cfg: DictConfig, key: str) -> float:
    """Read a required scalar from the domain-randomisation config.

    Raises :class:`ValueError` if the key is absent, for the same
    single-source-of-truth reason as :func:`_require_pair`.
    """
    val = OmegaConf.select(cfg, key, default=None)
    if val is None:
        raise ValueError(
            f"env.domain_randomization.{key} is required when domain randomisation is enabled."
        )
    return float(val)


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
        # configs/env.yaml is the single source of disturbance magnitudes. Every
        # range is REQUIRED when DR is enabled: a missing key raises rather than
        # silently falling back to a hardcoded default that could diverge from the
        # config. Set a channel to [0.0, 0.0] to leave it nominal.
        self._wind_mag = _require_pair(dr_cfg, "wind_magnitude_mps")
        self._mass = _require_pair(dr_cfg, "mass_offset_fraction")
        self._sigma = _require_pair(dr_cfg, "sensor_noise_sigma")
        self._spike_p = _require_pair(dr_cfg, "sensor_spike_probability")
        self._spike_mag = _require_scalar(dr_cfg, "sensor_spike_magnitude")
        delay = _require_pair(dr_cfg, "actuator_delay_steps")
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
