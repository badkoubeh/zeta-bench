"""Graduated disturbance-matrix runner — the primary robustness evaluation path.

Every controller faces the **same** graduated disturbance cells under the
**same** fixed seeds, so the resulting ``disturbance type × severity ×
success-rate`` table is reproducible and cross-comparable. This module owns the
reusable rollout + aggregation and the grid orchestration; the thin Hydra
entrypoint (``experiments/evaluate_robustness.py``) builds the controllers and
wires outputs (CSV, heatmap, optional wandb table).

Fairness invariants (see :mod:`robustness.disturbances`):

- **Identical conditions per cell.** For each (controller, cell) the per-episode
  seed stream is re-derived from the same master ``cfg.seed``, so every
  controller sees identical initial conditions *and* identical sensor-noise
  realisations within a cell.
- **Disturbance is the sole independent variable.** Initial conditions are
  pinned via a fixed curriculum schedule; ``task_difficulty`` never scales
  disturbances. A non-fixed schedule is flagged at runtime.

Import rule: may import from ``dynamics``/``envs``/``controllers``/``utils``;
never from ``experiments``.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from envs.rocket_landing_env import RocketLandingEnv
from robustness.disturbances import DisturbanceCell, iter_disturbance_cells
from utils.logging_config import get_logger
from utils.normalisation import FixedObsScaler

logger = get_logger(__name__)

# Per-cell descriptor columns (from the DisturbanceCell) followed by the summary
# metrics. ``wind_direction_deg`` / ``spike_probability`` are ``None`` for cells
# where the axis does not apply; they serialise to empty CSV fields.
CELL_COLUMNS: tuple[str, ...] = (
    "controller",
    "disturbance_type",
    "severity",
    "wind_direction_deg",
    "spike_probability",
    "label",
)
SUMMARY_COLUMNS: tuple[str, ...] = (
    "n_episodes",
    "success_rate",
    "n_success",
    "n_crash",
    "n_out_of_bounds",
    "n_timeout",
    "return_mean",
    "return_std",
    "touchdown_speed_mean_mps",
    "fuel_used_mean_kg",
    "episode_length_mean",
)
MATRIX_COLUMNS: tuple[str, ...] = CELL_COLUMNS + SUMMARY_COLUMNS


def run_episode(
    env: RocketLandingEnv,
    controller: object,
    scaler: FixedObsScaler,
    seed: int,
    initial_fuel_kg: float,
) -> dict[str, float | str | int]:
    """Run one episode to termination/truncation; return per-episode metrics.

    Uniform across controller types: every controller exposes
    ``predict(obs, deterministic=True)``. Controllers that carry integrator
    state (e.g. the PID baseline) also expose ``reset()``; it is called when
    present so stateful controllers start each episode clean.
    """
    reset_fn = getattr(controller, "reset", None)
    if callable(reset_fn):
        reset_fn()

    obs, _ = env.reset(seed=seed)
    ep_return = 0.0
    length = 0
    last_reason = "ongoing"

    while True:
        action = controller.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += float(reward)
        length += 1
        last_reason = str(info["termination_reason"])
        if terminated or truncated:
            break

    raw = scaler.unscale(obs)
    touchdown_speed_mps = float(np.linalg.norm(raw[3:6]))
    final_fuel_kg = float(raw[15])
    fuel_used_kg = float(initial_fuel_kg - final_fuel_kg)

    return {
        "seed": int(seed),
        "outcome": last_reason,
        "return": float(ep_return),
        "length": int(length),
        "touchdown_speed_mps": touchdown_speed_mps,
        "fuel_used_kg": fuel_used_kg,
    }


def summarise(rows: list[dict[str, float | str | int]]) -> dict[str, float | int]:
    """Aggregate per-episode rows into the per-cell summary metrics."""
    n = len(rows)
    if n == 0:
        return {
            "n_episodes": 0,
            "success_rate": 0.0,
            "n_success": 0,
            "n_crash": 0,
            "n_out_of_bounds": 0,
            "n_timeout": 0,
            "return_mean": 0.0,
            "return_std": 0.0,
            "touchdown_speed_mean_mps": 0.0,
            "fuel_used_mean_kg": 0.0,
            "episode_length_mean": 0.0,
        }

    outcomes = [str(r["outcome"]) for r in rows]
    returns = np.array([float(r["return"]) for r in rows], dtype=np.float64)
    speeds = np.array([float(r["touchdown_speed_mps"]) for r in rows], dtype=np.float64)
    fuels = np.array([float(r["fuel_used_kg"]) for r in rows], dtype=np.float64)
    lengths = np.array([int(r["length"]) for r in rows], dtype=np.float64)

    successes = outcomes.count("success")
    return {
        "n_episodes": n,
        "success_rate": successes / n,
        "n_success": successes,
        "n_crash": outcomes.count("crash"),
        "n_out_of_bounds": outcomes.count("out_of_bounds"),
        "n_timeout": outcomes.count("timeout"),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std(ddof=0)),
        "touchdown_speed_mean_mps": float(speeds.mean()),
        "fuel_used_mean_kg": float(fuels.mean()),
        "episode_length_mean": float(lengths.mean()),
    }


def _run_cell(
    env: RocketLandingEnv,
    controller: object,
    scaler: FixedObsScaler,
    master_seed: int,
    n_episodes: int,
    initial_fuel_kg: float,
) -> dict[str, float | int]:
    """Run all episodes of one (controller, cell) under a fixed seed stream.

    Re-deriving the per-episode seeds from ``master_seed`` here — rather than
    once globally — is what makes every controller face the identical seed
    sequence within a cell (the fairness invariant).
    """
    rng = np.random.default_rng(master_seed)
    ep_rows = []
    for _ in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        ep_rows.append(run_episode(env, controller, scaler, ep_seed, initial_fuel_kg))
    return summarise(ep_rows)


def run_matrix(
    cfg: DictConfig,
    controllers: dict[str, object],
) -> list[dict[str, object]]:
    """Evaluate every controller across the graduated disturbance matrix.

    For each cell from :func:`iter_disturbance_cells`, every controller is run
    for ``cfg.eval.seeds × cfg.eval.episodes_per_seed`` episodes under the same
    fixed seed stream. Returns one long-format row per (controller, cell) with
    the cell descriptor plus the summary metrics (columns :data:`MATRIX_COLUMNS`).

    Parameters
    ----------
    cfg : DictConfig
        Composed config with ``eval`` (grid), ``env``, ``seed`` sections.
    controllers : dict of str -> controller
        Controllers to compare, each exposing ``predict(obs, deterministic)``.
        Built by the entrypoint so this module stays free of model-loading /
        wandb concerns.
    """
    env = RocketLandingEnv(cfg)
    scaler = FixedObsScaler(cfg)
    initial_fuel_kg = float(cfg.env.dynamics.initial_fuel_kg)
    n_episodes = int(cfg.eval.seeds) * int(cfg.eval.episodes_per_seed)
    master_seed = int(cfg.seed)

    schedule = str(OmegaConf.select(cfg, "env.curriculum.schedule", default="linear"))
    if schedule != "fixed":
        logger.warning(
            "curriculum schedule is %r, not 'fixed' — initial conditions may drift "
            "across controllers/cells, breaking the identical-conditions fairness "
            "guarantee. Pin env.curriculum.schedule=fixed for the matrix.",
            schedule,
        )

    cells = list(iter_disturbance_cells(cfg.eval))
    logger.info(
        "matrix: %d cells × %d controllers × %d episodes = %d rollouts",
        len(cells),
        len(controllers),
        n_episodes,
        len(cells) * len(controllers) * n_episodes,
    )

    rows: list[dict[str, object]] = []
    for cell in cells:
        env.set_disturbance(**cell.disturbance.as_env_kwargs())
        for name, controller in controllers.items():
            summary = _run_cell(
                env, controller, scaler, master_seed, n_episodes, initial_fuel_kg
            )
            rows.append(_cell_row(name, cell, summary))
            logger.info(
                "cell=%-22s controller=%-4s success_rate=%.1f%% (%d/%d)",
                cell.label,
                name,
                100.0 * float(summary["success_rate"]),
                int(summary["n_success"]),
                int(summary["n_episodes"]),
            )
    return rows


def _cell_row(
    controller: str,
    cell: DisturbanceCell,
    summary: dict[str, float | int],
) -> dict[str, object]:
    """Assemble one long-format matrix row from a cell + its summary."""
    row: dict[str, object] = {
        "controller": controller,
        "disturbance_type": cell.disturbance_type,
        "severity": cell.severity,
        "wind_direction_deg": cell.wind_direction_deg,
        "spike_probability": cell.spike_probability,
        "label": cell.label,
    }
    row.update(summary)
    return row


def write_matrix_csv(rows: list[dict[str, object]], path: str | Path) -> Path:
    """Write matrix rows to ``path`` as CSV with the canonical column order."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MATRIX_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in MATRIX_COLUMNS})
    logger.info("wrote %s (%d rows)", out, len(rows))
    return out
