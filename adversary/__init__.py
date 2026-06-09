"""Learned disturbance adversary.

Zero-sum adversary that injects worst-case disturbances during training:
wind force, sensor noise scale, and payload mass offset. Trained as SB3 SAC
against ``-agent_reward``.

Update ratio (config-driven): ``N = 10`` agent gradient steps per
1 adversary step. Adversary weight is annealed from 0 → 1 via the curriculum
scheduler.

Import rule: this package may import from ``envs/``, ``dynamics/`` (read-only
types), and ``utils/``. It may be imported by ``experiments/``.
"""
from __future__ import annotations

from adversary.adversary_policy import (
    ADVERSARY_ACTION_DIM,
    ADVERSARY_OBS_DIM,
    AdversaryPolicy,
)

__all__ = ["ADVERSARY_ACTION_DIM", "ADVERSARY_OBS_DIM", "AdversaryPolicy"]
