"""Unit tests for :mod:`robustness.disturbances`.

Covers the typed :class:`Disturbance` value object, the polar-to-NED wind
helper, ``Disturbance.from_config``, and the graduated-matrix cell generator
:func:`iter_disturbance_cells` (cell taxonomy, severity tagging, nominal
deduplication, and the combined-at-max cell).
"""
from __future__ import annotations

import numpy as np
import pytest
from hydra import compose, initialize
from hypothesis import given, settings
from hypothesis import strategies as st
from omegaconf import OmegaConf

from robustness.disturbances import (
    Disturbance,
    DisturbanceCell,
    iter_disturbance_cells,
    wind_from_polar,
)


@pytest.fixture
def eval_cfg():
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="train")
    return cfg.eval


# --- Disturbance value object ---------------------------------------------

def test_none_is_nominal() -> None:
    """The default / ``none()`` disturbance reports as nominal on every axis."""
    d = Disturbance.none()
    assert d.is_nominal
    assert not d.has_wind
    assert not d.has_sensor_noise
    assert d == Disturbance()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"wind_velocity_ned": (1.0, 0.0, 0.0)},
        {"mass_offset_fraction": 0.1},
        {"sensor_noise_sigma": 0.05},
        {"sensor_spike_probability": 0.01},
        {"actuator_delay_steps": 2},
    ],
)
def test_any_active_field_is_not_nominal(kwargs) -> None:
    """A disturbance with any single non-zero field is not nominal."""
    assert not Disturbance(**kwargs).is_nominal


def test_wind_vector_ned_is_array_view() -> None:
    """``wind_vector_ned`` exposes the tuple field as a float64 array."""
    d = Disturbance(wind_velocity_ned=(2.0, -3.0, 0.0))
    vec = d.wind_vector_ned
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float64
    np.testing.assert_array_equal(vec, [2.0, -3.0, 0.0])
    assert d.has_wind


# --- wind_from_polar ------------------------------------------------------

@pytest.mark.parametrize(
    "direction_deg, expected",
    [
        (0.0, (1.0, 0.0, 0.0)),      # North -> +X
        (90.0, (0.0, 1.0, 0.0)),     # East  -> +Y
        (180.0, (-1.0, 0.0, 0.0)),   # South -> -X
        (270.0, (0.0, -1.0, 0.0)),   # West  -> -Y
    ],
)
def test_wind_from_polar_cardinals(direction_deg, expected) -> None:
    """Unit-magnitude wind maps compass bearings to the right NED axes."""
    got = wind_from_polar(1.0, direction_deg)
    np.testing.assert_allclose(got, expected, atol=1e-12)


@given(
    magnitude=st.floats(min_value=0.0, max_value=50.0),
    direction=st.floats(min_value=0.0, max_value=360.0),
)
@settings(max_examples=50, deadline=None)
def test_wind_from_polar_preserves_magnitude(magnitude, direction) -> None:
    """The horizontal wind vector always has the requested magnitude, Z = 0."""
    vec = np.array(wind_from_polar(magnitude, direction))
    assert vec[2] == 0.0
    assert np.isclose(np.linalg.norm(vec), magnitude, atol=1e-9)


# --- from_config ----------------------------------------------------------

def test_from_config_polar_wind_and_defaults() -> None:
    """``from_config`` reads polar wind and fills absent fields with nominal."""
    cfg = OmegaConf.create(
        {"wind_magnitude_mps": 5.0, "wind_direction_deg": 90.0, "mass_offset_fraction": -0.1}
    )
    d = Disturbance.from_config(cfg)
    np.testing.assert_allclose(d.wind_vector_ned, [0.0, 5.0, 0.0], atol=1e-12)
    assert d.mass_offset_fraction == -0.1
    assert d.sensor_noise_sigma == 0.0
    assert d.actuator_delay_steps == 0


def test_from_config_empty_is_nominal() -> None:
    """An empty config block yields the nominal disturbance."""
    assert Disturbance.from_config(OmegaConf.create({})).is_nominal


# --- iter_disturbance_cells -----------------------------------------------

def test_grid_starts_with_single_nominal_cell(eval_cfg) -> None:
    """The generator emits exactly one nominal cell, and it is first."""
    cells = list(iter_disturbance_cells(eval_cfg))
    nominal = [c for c in cells if c.disturbance_type == "nominal"]
    assert len(nominal) == 1
    assert cells[0].disturbance_type == "nominal"
    assert cells[0].disturbance.is_nominal
    assert cells[0].severity == 0.0


def test_grid_dedupes_zero_severity_against_nominal(eval_cfg) -> None:
    """Magnitude-0 wind and zero mass/noise never re-appear as their own cell."""
    cells = list(iter_disturbance_cells(eval_cfg))
    assert all(c.severity != 0.0 for c in cells if c.disturbance_type == "wind")
    assert all(c.severity != 0.0 for c in cells if c.disturbance_type == "mass")
    for c in cells:
        if c.disturbance_type == "sensor_noise":
            # A sensor-noise cell must perturb something.
            assert c.disturbance.has_sensor_noise


def test_grid_cell_counts_match_config(eval_cfg) -> None:
    """Cell counts equal the config sweep sizes (with zero-severity dropped)."""
    grid = eval_cfg.disturbance_grid
    cells = list(iter_disturbance_cells(eval_cfg))
    by_type = {t: [c for c in cells if c.disturbance_type == t] for t in
               ("nominal", "wind", "mass", "sensor_noise", "combined")}

    n_wind_mag = sum(1 for m in grid.wind.magnitudes_mps if float(m) != 0.0)
    n_dir = len(grid.wind.directions_deg)
    assert len(by_type["wind"]) == n_wind_mag * n_dir

    n_mass = sum(1 for m in grid.mass_offset_fraction if float(m) != 0.0)
    assert len(by_type["mass"]) == n_mass

    n_noise = sum(
        1
        for s in grid.sensor_noise.sigma
        for p in grid.sensor_noise.spike_probability
        if not (float(s) == 0.0 and float(p) == 0.0)
    )
    assert len(by_type["sensor_noise"]) == n_noise
    assert len(by_type["combined"]) == 1  # combined.enabled is true in the default grid


def test_wind_cells_carry_direction_and_severity(eval_cfg) -> None:
    """Wind cells expose magnitude as severity and record the bearing."""
    cells = [c for c in iter_disturbance_cells(eval_cfg) if c.disturbance_type == "wind"]
    for c in cells:
        assert c.wind_direction_deg is not None
        assert c.severity > 0.0
        assert np.isclose(np.linalg.norm(c.disturbance.wind_vector_ned), c.severity)


def test_combined_cell_activates_every_axis(eval_cfg) -> None:
    """The combined cell drives every disturbance type simultaneously at max."""
    grid = eval_cfg.disturbance_grid
    combined = next(
        c for c in iter_disturbance_cells(eval_cfg) if c.disturbance_type == "combined"
    )
    d = combined.disturbance
    assert d.has_wind
    assert d.mass_offset_fraction == max((float(m) for m in grid.mass_offset_fraction), key=abs)
    assert d.sensor_noise_sigma == max(float(s) for s in grid.sensor_noise.sigma)
    assert d.sensor_spike_probability == max(float(p) for p in grid.sensor_noise.spike_probability)
    assert np.isclose(np.linalg.norm(d.wind_vector_ned),
                      max(float(m) for m in grid.wind.magnitudes_mps))


def test_combined_cell_absent_when_disabled(eval_cfg) -> None:
    """Disabling the combined sweep removes the combined cell."""
    cfg = OmegaConf.merge(
        eval_cfg, OmegaConf.create({"disturbance_grid": {"combined": {"enabled": False}}})
    )
    cells = list(iter_disturbance_cells(cfg))
    assert not any(c.disturbance_type == "combined" for c in cells)


def test_all_cells_are_disturbance_cells(eval_cfg) -> None:
    """Every yielded item is a DisturbanceCell carrying a Disturbance."""
    for c in iter_disturbance_cells(eval_cfg):
        assert isinstance(c, DisturbanceCell)
        assert isinstance(c.disturbance, Disturbance)
        assert c.label
