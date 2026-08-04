"""Hydra entrypoint: end-to-end MPC evaluation.

Runs the model-predictive descent-guidance baseline through
:class:`envs.RocketLandingEnv` for ``cfg.eval_mpc.n_episodes`` episodes under
nominal conditions (no wind, mass offset, or sensor noise — those belong to the
graduated matrix in ``experiments/evaluate_robustness.py``). Writes per-episode
rows and an aggregate summary to disk and prints a human-readable summary.

This is the loop for tuning ``configs/mpc_controller.yaml``. Tune here, under
nominal conditions; never against robustness-matrix cells.

Alongside the landing metrics it reports **solver cost** — mean and p99
milliseconds per solve, and solves per episode. The full disturbance matrix is
millions of control ticks, so the affordability of a horizon/cadence setting
should be a measured number before a long run is launched, not a hope.

CLI examples
------------
    python experiments/evaluate_mpc.py
    python experiments/evaluate_mpc.py seed=7 eval_mpc.n_episodes=20
    python experiments/evaluate_mpc.py mpc_controller.envelope.margin=0.4
    python experiments/evaluate_mpc.py eval_mpc.render=true

Outputs
-------
- ``results/{run_name}/episodes.csv`` — one row per episode.
- ``results/{run_name}/summary.json`` — aggregates across episodes, including
  the solver-cost fields.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from controllers.mpc_baseline import MPCController
from envs.rocket_landing_env import RocketLandingEnv
from robustness.evaluation import summarise
from utils.logging_config import get_logger
from utils.normalisation import FixedObsScaler
from utils.render import TrajectoryBuffer, animate_side_view, plot_timeseries

logger = get_logger(__name__)

_EPISODE_COLUMNS: tuple[str, ...] = (
    "episode_idx",
    "seed",
    "outcome",
    "return",
    "length",
    "touchdown_speed_mps",
    "fuel_used_kg",
    "n_solves",
    "solve_ms_mean",
    "solve_ms_p99",
)


def _run_episode(
    env: RocketLandingEnv,
    controller: MPCController,
    scaler: FixedObsScaler,
    seed: int,
    initial_fuel_kg: float,
    buffer: TrajectoryBuffer | None = None,
) -> dict[str, float | str | int]:
    """Run one episode to termination/truncation; return per-episode metrics.

    Mirrors :func:`robustness.evaluation.run_episode` (the matrix path) and adds
    two things it deliberately has no business carrying: optional trajectory
    recording for the renderer, and the controller's per-episode solve
    telemetry.
    """
    controller.reset()
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
        if buffer is not None:
            buffer.append(obs, action, float(reward))
        if terminated or truncated:
            break

    raw = scaler.unscale(obs)
    touchdown_speed_mps = float(np.linalg.norm(raw[3:6]))
    final_fuel_kg = float(raw[15])

    row: dict[str, float | str | int] = {
        "seed": int(seed),
        "outcome": last_reason,
        "return": float(ep_return),
        "length": int(length),
        "touchdown_speed_mps": touchdown_speed_mps,
        "fuel_used_kg": float(initial_fuel_kg - final_fuel_kg),
    }
    row.update(controller.solve_stats())
    return row


def _render_best_and_worst(
    cfg: DictConfig,
    rows: list[dict[str, float | str | int]],
    buffers: list[TrajectoryBuffer | None],
    results_dir: Path,
) -> None:
    """Render time-series PNG + side-view MP4 for the best and worst episodes."""
    if not rows:
        return

    returns = [float(r["return"]) for r in rows]
    selected = sorted({int(np.argmax(returns)), int(np.argmin(returns))})

    plots_dir = results_dir / "plots"
    video_dir = results_dir / "video"
    plots_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    fps = int(cfg.eval_mpc.get("render_fps", int(cfg.env.episode.control_hz)))
    scene_meta = {
        "pad_radius_m": float(cfg.env.touchdown.get("pad_radius_m", 30.0)),
        "oob_cylinder_radius_m": float(cfg.env.oob.cylinder_radius_m),
        "oob_ceiling_m": float(cfg.env.oob.ceiling_m),
        # The MPC has no constant descent target — its reference is the
        # altitude-dependent envelope. Draw the terminal target so the velocity
        # panel still carries a reference line.
        "target_descent_mps": float(cfg.mpc_controller.envelope.touchdown_target_mps),
    }

    for idx in selected:
        buf = buffers[idx]
        if buf is None:
            continue
        row = rows[idx]
        meta = dict(scene_meta)
        meta.update(
            {
                "outcome": str(row["outcome"]),
                "episode_idx": int(row["episode_idx"]),
                "seed": int(row["seed"]),
                "return_total": float(row["return"]),
            }
        )
        traj = buf.finalize(meta)
        tag = f"ep{int(row['episode_idx']):02d}_{row['outcome']}"
        png_path = plots_dir / f"timeseries_{tag}.png"
        mp4_path = video_dir / f"landing_{tag}.mp4"
        logger.info("rendering %s", png_path)
        plot_timeseries(traj, png_path)
        logger.info("rendering %s (this may take a minute)", mp4_path)
        animate_side_view(traj, mp4_path, fps=fps)


def _solver_cost(rows: list[dict[str, float | str | int]]) -> dict[str, float]:
    """Aggregate the per-episode solve telemetry into run-level solver cost."""
    if not rows:
        return {"solves_per_episode_mean": 0.0, "solve_ms_mean": 0.0, "solve_ms_p99_max": 0.0}
    solves = np.array([float(r["n_solves"]) for r in rows], dtype=np.float64)
    means = np.array([float(r["solve_ms_mean"]) for r in rows], dtype=np.float64)
    p99s = np.array([float(r["solve_ms_p99"]) for r in rows], dtype=np.float64)
    return {
        "solves_per_episode_mean": float(solves.mean()),
        # Weight each episode's mean by its solve count so the run-level mean is
        # the true per-solve cost, not a mean of means over uneven episodes.
        "solve_ms_mean": float(np.average(means, weights=solves)) if solves.sum() else 0.0,
        "solve_ms_p99_max": float(p99s.max()),
    }


@hydra.main(config_path="../configs", config_name="eval_mpc", version_base=None)
def main(cfg: DictConfig) -> None:
    """Build env + MPC, run ``n_episodes``, dump CSV + JSON + stdout summary."""
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("run_name=%s results_dir=%s", cfg.run_name, results_dir)
    logger.info(
        "n_episodes=%d seed=%d task_difficulty=%.3f",
        int(cfg.eval_mpc.n_episodes),
        int(cfg.seed),
        float(cfg.eval_mpc.task_difficulty),
    )
    mpc = cfg.mpc_controller
    logger.info(
        "mpc: horizon=%d × %.3f s = %.2f s window, replan every %d ticks, margin=%.2f",
        int(mpc.horizon),
        int(mpc.step_ticks) / float(cfg.env.episode.control_hz),
        int(mpc.horizon) * int(mpc.step_ticks) / float(cfg.env.episode.control_hz),
        int(mpc.resolve_every_n_ticks),
        float(mpc.envelope.margin),
    )

    env = RocketLandingEnv(cfg)
    controller = MPCController(cfg)
    scaler = FixedObsScaler(cfg)
    initial_fuel_kg = float(cfg.env.dynamics.initial_fuel_kg)
    control_dt = 1.0 / float(cfg.env.episode.control_hz)

    render_enabled = bool(cfg.eval_mpc.get("render", False))

    rng = np.random.default_rng(int(cfg.seed))
    rows: list[dict[str, float | str | int]] = []
    buffers: list[TrajectoryBuffer | None] = []
    for ep_idx in range(int(cfg.eval_mpc.n_episodes)):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        buffer = TrajectoryBuffer(control_dt, scaler) if render_enabled else None
        row = _run_episode(env, controller, scaler, ep_seed, initial_fuel_kg, buffer=buffer)
        row["episode_idx"] = ep_idx
        rows.append(row)
        buffers.append(buffer)
        logger.info(
            "ep %02d/%02d seed=%d outcome=%-13s return=%9.2f len=%4d v_td=%5.2f m/s "
            "fuel=%6.1f kg solves=%4d @ %.2f ms",
            ep_idx + 1,
            int(cfg.eval_mpc.n_episodes),
            ep_seed,
            row["outcome"],
            row["return"],
            row["length"],
            row["touchdown_speed_mps"],
            row["fuel_used_kg"],
            int(row["n_solves"]),
            row["solve_ms_mean"],
        )

    if render_enabled:
        _render_best_and_worst(cfg, rows, buffers, results_dir)

    episodes_path = results_dir / "episodes.csv"
    summary_path = results_dir / "summary.json"

    with episodes_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_EPISODE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in _EPISODE_COLUMNS})

    summary = summarise(rows)
    summary.update(_solver_cost(rows))
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    logger.info("wrote %s", episodes_path)
    logger.info("wrote %s", summary_path)
    logger.info(
        "summary: success_rate=%.2f%% (%d/%d)  return=%.2f ± %.2f  v_td_mean=%.2f m/s",
        100.0 * summary["success_rate"],
        summary["n_success"],
        summary["n_episodes"],
        summary["return_mean"],
        summary["return_std"],
        summary["touchdown_speed_mean_mps"],
    )
    logger.info(
        "solver: %.0f solves/episode  %.2f ms mean  %.2f ms p99",
        summary["solves_per_episode_mean"],
        summary["solve_ms_mean"],
        summary["solve_ms_p99_max"],
    )


if __name__ == "__main__":
    main()
