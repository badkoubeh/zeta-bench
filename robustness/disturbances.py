"""Typed, composable disturbance models for the graduated robustness matrix.

A :class:`Disturbance` is a single, immutable *value object* bundling the four
disturbance types ZetaBench characterises. Every field defaults to zero, so the
default instance is the **nominal** (no-disturbance) condition. Each type is
independently testable AND composable — several can be active at once (the
``combined`` cell exercises them simultaneously).

Where each type is applied (see the wiring in :mod:`dynamics` / :mod:`envs`):

- **wind** — a horizontal air-mass velocity (m/s, NED). Enters the physics via
  the drag term as relative airspeed ``v_air = v_rocket − v_wind`` (physically
  honest; matches the m/s units of the config grid).
- **mass uncertainty** — a signed fraction offsetting the vehicle dry mass;
  applied by the env rebuilding the dynamics parameters.
- **sensor noise** — Gaussian σ plus sparse spikes, added to the *scaled*
  observation (σ is a fraction of each sensor's full-scale range).
- **actuator delay** — an integer control-tick latency on the applied action.

``disturbance_severity`` (the term used across the project) is the graduated
magnitude that indexes a cell along its axis: wind magnitude (m/s), the signed
mass-offset fraction, or the sensor-noise σ. See :func:`iter_disturbance_cells`.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig


@dataclass(frozen=True)
class Disturbance:
    """An immutable, composable disturbance configuration.

    All fields default to the nominal (zero) value. Instances are hashable value
    objects — ``wind_velocity_ned`` is stored as a plain tuple so equality and
    hashing behave, with :attr:`wind_vector_ned` exposing the numpy view the
    dynamics consume.

    Parameters
    ----------
    wind_velocity_ned : tuple of 3 floats
        Air-mass velocity in the NED inertial frame (m/s). Built from a polar
        (magnitude, bearing) via :func:`wind_from_polar`.
    mass_offset_fraction : float
        Signed fraction offsetting the vehicle dry mass (e.g. ``+0.20`` = 20 %
        heavier than nominal, ``−0.10`` = 10 % lighter).
    sensor_noise_sigma : float
        Standard deviation of zero-mean Gaussian noise added to the scaled
        observation (fraction of full scale).
    sensor_spike_probability : float
        Per-component, per-step probability of an additive outlier spike.
    sensor_spike_magnitude : float
        Magnitude of an outlier spike (scaled-observation units). Only relevant
        when ``sensor_spike_probability > 0``.
    actuator_delay_steps : int
        Number of control ticks by which the applied action lags the command.
    """

    wind_velocity_ned: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mass_offset_fraction: float = 0.0
    sensor_noise_sigma: float = 0.0
    sensor_spike_probability: float = 0.0
    sensor_spike_magnitude: float = 0.0
    actuator_delay_steps: int = 0

    @classmethod
    def none(cls) -> Disturbance:
        """Return the nominal disturbance (all fields zero)."""
        return cls()

    @classmethod
    def from_config(cls, dist_cfg: DictConfig) -> Disturbance:
        """Build a disturbance from an ``env.disturbance`` config block.

        Wind is specified in polar form (``wind_magnitude_mps`` +
        ``wind_direction_deg``) and converted to a NED velocity via
        :func:`wind_from_polar`. Any absent field falls back to its nominal
        (zero) default, so a partial or empty block yields a valid disturbance.
        """
        return cls(
            wind_velocity_ned=wind_from_polar(
                float(dist_cfg.get("wind_magnitude_mps", 0.0)),
                float(dist_cfg.get("wind_direction_deg", 0.0)),
            ),
            mass_offset_fraction=float(dist_cfg.get("mass_offset_fraction", 0.0)),
            sensor_noise_sigma=float(dist_cfg.get("sensor_noise_sigma", 0.0)),
            sensor_spike_probability=float(dist_cfg.get("sensor_spike_probability", 0.0)),
            sensor_spike_magnitude=float(dist_cfg.get("sensor_spike_magnitude", 0.0)),
            actuator_delay_steps=int(dist_cfg.get("actuator_delay_steps", 0)),
        )

    @property
    def wind_vector_ned(self) -> NDArray[np.float64]:
        """Wind air-mass velocity as a ``(3,)`` float64 array."""
        return np.array(self.wind_velocity_ned, dtype=np.float64)

    @property
    def has_wind(self) -> bool:
        """True when any wind component is non-zero."""
        return any(w != 0.0 for w in self.wind_velocity_ned)

    @property
    def has_sensor_noise(self) -> bool:
        """True when Gaussian noise or spikes are active."""
        return self.sensor_noise_sigma > 0.0 or self.sensor_spike_probability > 0.0

    @property
    def is_nominal(self) -> bool:
        """True when no disturbance of any kind is active."""
        return (
            not self.has_wind
            and self.mass_offset_fraction == 0.0
            and not self.has_sensor_noise
            and self.actuator_delay_steps == 0
        )

    def as_env_kwargs(self) -> dict[str, object]:
        """Unpack into the primitive kwargs of ``RocketLandingEnv.set_disturbance``.

        This is the layering seam: the robustness layer owns the mapping from a
        typed ``Disturbance`` to the env's primitive hook, so the env never has
        to import ``robustness``. Wind is passed as ``None`` when nominal so the
        drag term skips the relative-airspeed subtraction entirely.
        """
        return {
            "wind_velocity_ned": self.wind_vector_ned if self.has_wind else None,
            "mass_offset_fraction": self.mass_offset_fraction,
            "sensor_noise_sigma": self.sensor_noise_sigma,
            "sensor_spike_probability": self.sensor_spike_probability,
            "sensor_spike_magnitude": self.sensor_spike_magnitude,
            "actuator_delay_steps": self.actuator_delay_steps,
        }


@dataclass(frozen=True)
class DisturbanceCell:
    """One cell of the graduated matrix: a disturbance plus its axis metadata.

    Parameters
    ----------
    disturbance_type : str
        One of ``"nominal" | "wind" | "mass" | "sensor_noise" | "combined"``.
    severity : float
        The graduated magnitude indexing this cell along its axis
        (``disturbance_severity``): wind magnitude (m/s), the signed mass-offset
        fraction, or the sensor-noise σ. ``0.0`` for the nominal cell; for the
        combined cell it is the maximum wind magnitude used.
    disturbance : Disturbance
        The disturbance to apply for this cell.
    label : str
        Human-readable identifier, unique across the grid.
    wind_direction_deg : float or None
        Compass bearing for wind cells (``None`` otherwise); retained as a
        separate CSV column so directional weakness stays visible.
    spike_probability : float or None
        Spike probability for sensor-noise cells (``None`` otherwise).
    """

    disturbance_type: str
    severity: float
    disturbance: Disturbance
    label: str
    wind_direction_deg: float | None = None
    spike_probability: float | None = None


def wind_from_polar(magnitude_mps: float, direction_deg: float) -> tuple[float, float, float]:
    """Build a horizontal NED wind velocity from a polar (magnitude, bearing).

    The returned vector is the **air-mass velocity** pointing along the compass
    bearing in the NED horizontal plane (X = North, Y = East, Z = Down):

    - ``0°``   → ``(+mag, 0, 0)`` (North)
    - ``90°``  → ``(0, +mag, 0)`` (East)
    - ``180°`` → ``(−mag, 0, 0)`` (South)
    - ``270°`` → ``(0, −mag, 0)`` (West)

    Vertical wind is out of scope for the moderate-fidelity model, so the Z
    component is always zero.
    """
    theta = np.deg2rad(float(direction_deg))
    magnitude = float(magnitude_mps)
    return (magnitude * float(np.cos(theta)), magnitude * float(np.sin(theta)), 0.0)


def iter_disturbance_cells(eval_cfg: DictConfig) -> Iterator[DisturbanceCell]:
    """Yield the graduated disturbance-matrix cells from ``cfg.eval``.

    Reads ``eval_cfg.disturbance_grid`` and produces, in order:

    1. the nominal cell (baseline);
    2. one cell per (wind magnitude × direction), skipping magnitude 0 (which is
       the nominal cell);
    3. one cell per non-zero mass-offset fraction;
    4. one cell per (sensor-noise σ × spike probability), skipping the all-zero
       combination (nominal);
    5. a single ``combined`` cell with every disturbance at its maximum level,
       when ``disturbance_grid.combined.enabled`` is true.

    Each cell carries its ``disturbance_type`` and ``severity``
    (``disturbance_severity``) so the matrix runner and heatmap can index it. The
    nominal cell appears exactly once — magnitude-0 wind and zero mass/noise are
    deduplicated against it.
    """
    grid = eval_cfg.disturbance_grid

    yield DisturbanceCell(
        disturbance_type="nominal",
        severity=0.0,
        disturbance=Disturbance.none(),
        label="nominal",
    )

    # 2. Wind: magnitude × direction (magnitude 0 == nominal, skip it).
    for magnitude in grid.wind.magnitudes_mps:
        mag = float(magnitude)
        if mag == 0.0:
            continue
        for direction in grid.wind.directions_deg:
            deg = float(direction)
            yield DisturbanceCell(
                disturbance_type="wind",
                severity=mag,
                disturbance=Disturbance(wind_velocity_ned=wind_from_polar(mag, deg)),
                label=f"wind_{mag:g}mps_{deg:g}deg",
                wind_direction_deg=deg,
            )

    # 3. Mass offset (0.0 == nominal, skip it).
    for offset in grid.mass_offset_fraction:
        frac = float(offset)
        if frac == 0.0:
            continue
        yield DisturbanceCell(
            disturbance_type="mass",
            severity=frac,
            disturbance=Disturbance(mass_offset_fraction=frac),
            label=f"mass_{frac:+g}",
        )

    # 4. Sensor noise: sigma × spike probability (all-zero == nominal, skip it).
    spike_magnitude = float(grid.sensor_noise.spike_magnitude)
    for sigma in grid.sensor_noise.sigma:
        sig = float(sigma)
        for spike in grid.sensor_noise.spike_probability:
            spk = float(spike)
            if sig == 0.0 and spk == 0.0:
                continue
            yield DisturbanceCell(
                disturbance_type="sensor_noise",
                severity=sig,
                disturbance=Disturbance(
                    sensor_noise_sigma=sig,
                    sensor_spike_probability=spk,
                    sensor_spike_magnitude=spike_magnitude,
                ),
                label=f"noise_sig{sig:g}_spk{spk:g}",
                spike_probability=spk,
            )

    # 5. Combined cell — every disturbance at its maximum level.
    if bool(grid.combined.enabled):
        max_wind = max(float(m) for m in grid.wind.magnitudes_mps)
        combined_dir = float(grid.wind.directions_deg[0])
        max_mass = max((float(m) for m in grid.mass_offset_fraction), key=abs)
        max_sigma = max(float(s) for s in grid.sensor_noise.sigma)
        max_spike = max(float(p) for p in grid.sensor_noise.spike_probability)
        yield DisturbanceCell(
            disturbance_type="combined",
            severity=max_wind,
            disturbance=Disturbance(
                wind_velocity_ned=wind_from_polar(max_wind, combined_dir),
                mass_offset_fraction=max_mass,
                sensor_noise_sigma=max_sigma,
                sensor_spike_probability=max_spike,
                sensor_spike_magnitude=spike_magnitude,
            ),
            label="combined_max",
            wind_direction_deg=combined_dir,
            spike_probability=max_spike,
        )
