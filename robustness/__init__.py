"""Robustness layer — the ZetaBench product.

Typed, composable disturbance models and the graduated evaluation matrix that
stress-tests any controller across a fixed-seed disturbance grid, producing the
signature ``disturbance type × severity × success-rate`` heatmap.

Vocabulary
----------
``disturbance_severity`` is the knob that injects an external disturbance into
the system so its effect on robustness can be measured. It is a **separate axis**
from ``task_difficulty`` (which only scales the nominal initial-condition
envelope — see :mod:`envs.curriculum`). The graduated matrix is the *primary*,
cross-comparable mode; the adversarial/worst-case search (``adversary/``) is a
secondary stress mode whose results are reported separately and never merged into
the comparable matrix.

Import rule: this package may import from ``dynamics/``, ``envs/``,
``controllers/`` and ``utils/``; it must not import from ``experiments/``.
"""
from __future__ import annotations

from robustness.disturbances import (
    Disturbance,
    DisturbanceCell,
    iter_disturbance_cells,
    wind_from_polar,
)

__all__ = [
    "Disturbance",
    "DisturbanceCell",
    "iter_disturbance_cells",
    "wind_from_polar",
]
