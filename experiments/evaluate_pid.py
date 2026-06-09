"""Hydra entrypoint: end-to-end PID evaluation.

Runs the cascaded PID baseline through :class:`envs.RocketLandingEnv` for
``cfg.eval_pid.n_episodes`` episodes under nominal conditions (no wind,
mass offset, or sensor noise — those land with the Phase 3 robustness
sweep). Writes per-episode rows and an aggregate summary to disk and
prints a human-readable summary to stdout.

CLI examples
------------
    python experiments/evaluate_pid.py
    python experiments/evaluate_pid.py seed=7 eval_pid.n_episodes=20
    python experiments/evaluate_pid.py eval_pid.curriculum_progress=0.5

Outputs
-------
- ``results/{run_name}/episodes.csv`` — one row per episode.
- ``results/{run_name}/summary.json`` — aggregates across episodes.

The PID controller's gains live in ``configs/pid_controller.yaml``;
re-run after editing to evaluate new gains.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import hydra
import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from controllers.pid_baseline import PIDController
from envs.rocket_landing_env import RocketLandingEnv
from utils.logging_config import get_logger
from utils.normalisation import FixedObsScaler
from utils.render import Trajectory, animate_side_view, plot_timeseries

logger = get_logger(__name__)

_EPISODE_COLUMNS: tuple[str, ...] = (
    "episode_idx",
    "seed",
    "outcome",
    "return",
    "length",
    "touchdown_speed_mps",
    "fuel_used_kg",
)


class _TrajectoryBuffer:
    """Accumulate per-step physical-unit state for one episode's trajectory.

    The episode loop calls :meth:`append` after each ``env.step``. Once
    the episode terminates, :meth:`finalize` returns a :class:`Trajectory`
    that the renderer can consume.
    """

    def __init__(self, dt: float, scaler: FixedObsScaler) -> None:
        self._dt = float(dt)
        self._scaler = scaler
        self._obs_raw: list[NDArray[np.float64]] = []
        self._action: list[NDArray[np.float64]] = []
        self._reward: list[float] = []

    def append(
        self,
        obs_scaled: NDArray[np.float64],
        action: NDArray[np.float64],
        reward: float,
    ) -> None:
        self._obs_raw.append(self._scaler.unscale(obs_scaled).copy())
        self._action.append(np.asarray(action, dtype=np.float64).copy())
        self._reward.append(float(reward))

    def finalize(self, meta: dict) -> Trajectory:
        raw = np.stack(self._obs_raw, axis=0)
        action_arr = np.stack(self._action, axis=0)
        reward_arr = np.array(self._reward, dtype=np.float64)
        T = raw.shape[0]
        return Trajectory(
            t=np.arange(T, dtype=np.float64) * self._dt,
            pos_NED=raw[:, 0:3],
            vel_NED=raw[:, 3:6],
            euler=raw[:, 6:9],
            omega_body=raw[:, 9:12],
            action=action_arr,
            reward=reward_arr,
            fuel_kg=raw[:, 15],
            meta=meta,
        )


def _run_episode(
    env: RocketLandingEnv,
    controller: PIDController,
    scaler: FixedObsScaler,
    seed: int,
    initial_fuel_kg: float,
    buffer: _TrajectoryBuffer | None = None,
) -> dict[str, float | str | int]:
    """Run one episode to termination/truncation; return per-episode metrics.

    If ``buffer`` is provided, per-step state is appended to it (used for
    post-hoc rendering).
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
    velocity_NED = raw[3:6]
    touchdown_speed_mps = float(np.linalg.norm(velocity_NED))
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


def _summarise(rows: list[dict[str, float | str | int]]) -> dict[str, float | int]:
    """Aggregate per-episode rows into top-line stats."""
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


def _render_best_and_worst(
    cfg: DictConfig,
    rows: list[dict[str, float | str | int]],
    buffers: list[_TrajectoryBuffer | None],
    results_dir: Path,
) -> None:
    """Render time-series PNG + side-view MP4 for the best and worst episodes."""
    if not rows:
        return

    returns = [float(r["return"]) for r in rows]
    best_idx = int(np.argmax(returns))
    worst_idx = int(np.argmin(returns))
    selected = sorted({best_idx, worst_idx})

    plots_dir = results_dir / "plots"
    video_dir = results_dir / "video"
    plots_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    fps = int(cfg.eval_pid.get("render_fps", int(cfg.env.episode.control_hz)))
    target_descent_mps = float(cfg.pid_controller.altitude.target_descent_mps)
    scene_meta = {
        "pad_radius_m": float(cfg.env.touchdown.get("pad_radius_m", 30.0)),
        "oob_cylinder_radius_m": float(cfg.env.oob.cylinder_radius_m),
        "oob_ceiling_m": float(cfg.env.oob.ceiling_m),
        "target_descent_mps": target_descent_mps,
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


@hydra.main(config_path="../configs", config_name="eval_pid", version_base=None)
def main(cfg: DictConfig) -> None:
    """Build env + PID, run ``n_episodes``, dump CSV + JSON + stdout summary."""
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("run_name=%s results_dir=%s", cfg.run_name, results_dir)
    logger.info(
        "n_episodes=%d seed=%d curriculum_progress=%.3f",
        int(cfg.eval_pid.n_episodes),
        int(cfg.seed),
        float(cfg.eval_pid.curriculum_progress),
    )

    env = RocketLandingEnv(cfg)
    controller = PIDController(cfg)
    scaler = FixedObsScaler(cfg)
    initial_fuel_kg = float(cfg.env.dynamics.initial_fuel_kg)
    control_dt = 1.0 / float(cfg.env.episode.control_hz)

    render_enabled = bool(cfg.eval_pid.get("render", False))

    rng = np.random.default_rng(int(cfg.seed))
    rows: list[dict[str, float | str | int]] = []
    buffers: list[_TrajectoryBuffer | None] = []
    for ep_idx in range(int(cfg.eval_pid.n_episodes)):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        buffer = _TrajectoryBuffer(control_dt, scaler) if render_enabled else None
        row = _run_episode(env, controller, scaler, ep_seed, initial_fuel_kg, buffer=buffer)
        row["episode_idx"] = ep_idx
        rows.append(row)
        buffers.append(buffer)
        logger.info(
            "ep %02d/%02d seed=%d outcome=%-13s return=%9.2f len=%4d v_td=%5.2f m/s fuel=%6.1f kg",
            ep_idx + 1,
            int(cfg.eval_pid.n_episodes),
            ep_seed,
            row["outcome"],
            row["return"],
            row["length"],
            row["touchdown_speed_mps"],
            row["fuel_used_kg"],
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

    summary = _summarise(rows)
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


if __name__ == "__main__":
    main()
