"""Gymnasium environment wrapper for rocket landing.

Consumes :mod:`dynamics`. Exposes a single env ID ``RocketLanding-v0`` whose
behaviour (fidelity, curriculum stage, adversary on/off, sensor noise) is
parametrised at construction via a Hydra ``DictConfig``.

Import rule: this package may import from ``dynamics/`` and ``utils/``. It may
be imported by ``controllers/``, ``adversary/``, and ``experiments/``.
"""
from __future__ import annotations

from gymnasium.envs.registration import register

from envs.rocket_landing_env import RocketLandingEnv

register(
    id="RocketLanding-v0",
    entry_point="envs.rocket_landing_env:RocketLandingEnv",
    max_episode_steps=1500,
)

__all__ = ["RocketLandingEnv"]
