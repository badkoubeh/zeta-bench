"""Adversary policy: learned disturbance injector.

Observation (20-dim)
--------------------
Agent's 17-dim observation concatenated with the agent's last
3-dim action. Lets the adversary anticipate agent intent without
seeing hidden environment state.

Action (5-dim)
--------------
    [0:3] wind_force_NED        scaled by ±wind_force_max_N
    [3]   noise_magnitude       scaled to [0, noise_magnitude_max]
    [4]   mass_offset_fraction  scaled by ±mass_offset_max_fraction

Reward: ``−agent_reward`` (zero-sum).
"""
from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

ADVERSARY_OBS_DIM: int = 20
ADVERSARY_ACTION_DIM: int = 5


class AdversaryPolicy:
    """SB3 SAC adversary that injects disturbances into the agent's env."""

    def __init__(self, cfg: DictConfig) -> None:
        """Construct from the adversary config."""
        self._cfg = cfg

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Compute a 5-dim disturbance from a 20-dim observation."""
        raise NotImplementedError

    def update(self, batch: object) -> dict[str, float]:
        """Perform one adversary gradient step; return scalar metrics for wandb."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Persist the adversary SB3 model to disk."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "AdversaryPolicy":
        """Restore adversary from saved checkpoint."""
        raise NotImplementedError
