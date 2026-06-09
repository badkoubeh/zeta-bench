"""Unit tests for :class:`utils.normalisation.FixedObsScaler`."""
from __future__ import annotations

import numpy as np
import pytest
from hydra import compose, initialize

from utils.normalisation import OBS_DIM, FixedObsScaler


@pytest.fixture
def cfg():
    with initialize(config_path="../configs", version_base=None):
        return compose(config_name="train")


def test_scaler_zero_in_zero_out(cfg) -> None:
    """Scaling a zero vector returns a zero vector regardless of bounds."""
    scaler = FixedObsScaler(cfg)
    assert np.allclose(scaler.scale(np.zeros(OBS_DIM)), 0.0)


def test_scaler_round_trip(cfg) -> None:
    """``unscale(scale(x)) == x`` for a non-trivial vector across all slots."""
    scaler = FixedObsScaler(cfg)
    # Pick values that aren't symmetric to catch sign / direction errors
    x = np.array(
        [
            100.0, -50.0, 300.0,  # position xy, z
            10.0, -5.0, 20.0,     # velocity
            0.5, -0.3, 1.2,       # euler
            0.1, -0.2, 0.05,      # angular rate
            0.7, 0.2, -0.4,       # commanded action (passthrough)
            2500.0,               # fuel mass
            0.5,                  # fuel fraction (passthrough)
        ],
        dtype=np.float64,
    )
    assert np.allclose(scaler.unscale(scaler.scale(x)), x)


def test_scaler_position_z_matches_configured_bound(cfg) -> None:
    """A position-z equal to the configured bound scales to exactly 1.0."""
    scaler = FixedObsScaler(cfg)
    x = np.zeros(OBS_DIM)
    x[2] = cfg.env.obs_scaler.position_z_m
    out = scaler.scale(x)
    assert np.isclose(out[2], 1.0)


def test_scaler_passthrough_slots_unchanged(cfg) -> None:
    """Slots 12-14 (commanded action) and slot 16 (fuel fraction) have
    bound = 1, so scaling and unscaling are identity for them.
    """
    scaler = FixedObsScaler(cfg)
    x = np.zeros(OBS_DIM)
    x[12] = 0.8
    x[13] = -0.3
    x[14] = 0.5
    x[16] = 0.6
    out = scaler.scale(x)
    assert out[12] == 0.8
    assert out[13] == -0.3
    assert out[14] == 0.5
    assert out[16] == 0.6
