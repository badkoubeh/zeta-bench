"""Seeded RNG factory.

Always use :func:`make_rng` rather than ``np.random.seed(...)``. The returned
:class:`numpy.random.Generator` is independent of the global RNG so callers
can hold their own deterministic streams without cross-talk.
"""
from __future__ import annotations

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Return a deterministically seeded :class:`numpy.random.Generator`.

    Parameters
    ----------
    seed : int
        Integer seed. Two calls with the same ``seed`` produce identical
        sample streams.

    Returns
    -------
    numpy.random.Generator
        Local PCG64-backed generator.
    """
    return np.random.default_rng(seed)
