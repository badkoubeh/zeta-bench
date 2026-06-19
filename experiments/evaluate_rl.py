"""Hydra entrypoint: end-to-end RL agent evaluation (SAC / PPO).

Loads a trained model — either from the wandb Model Registry or a local .zip
path — and runs it through :class:`envs.RocketLandingEnv` for
``cfg.eval_rl.n_episodes`` episodes. Writes per-episode rows and an aggregate
summary to disk and prints a human-readable summary to stdout.

CLI examples
------------
    # from wandb Model Registry (canonical)
    python experiments/evaluate_rl.py agent=sac \\
      eval_rl.model_artifact="entity/project/zetabench-sac:best"

    # local .zip path (offline / no wandb)
    python experiments/evaluate_rl.py agent=sac \\
      eval_rl.model_path=results/sac_moderate_nominal_42/best_model.zip

    # PPO, 200 episodes, full difficulty
    python experiments/evaluate_rl.py agent=ppo \\
      eval_rl.model_artifact="entity/project/zetabench-ppo:best" \\
      eval_rl.n_episodes=200

    # Wind disturbance test (Phase 3 placeholder)
    python experiments/evaluate_rl.py agent=sac \\
      eval_rl.model_artifact="entity/project/zetabench-sac:best" \\
      eval_rl.disturbance.wind_mps=5.0

Model resolution
----------------
``eval_rl.model_artifact`` takes precedence. When set, the artifact is
downloaded via ``wandb.use_artifact`` and the local ``model.zip`` path is used.
Falls back to ``eval_rl.model_path`` (local file) when artifact is null.

Outputs
-------
- ``results/{run_name}/episodes.csv`` — one row per episode.
- ``results/{run_name}/summary.json`` — aggregates across episodes.
- ``results/{run_name}/plots/`` + ``results/{run_name}/video/`` — optional renders.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import hydra
from dotenv import load_dotenv

load_dotenv()
import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

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

_AGENTS = {"sac": "controllers.sac_agent.SACAgent", "ppo": "controllers.ppo_agent.PPOAgent"}


class _TrajectoryBuffer:
    """Accumulate per-step physical-unit state for one episode's trajectory."""

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


def _resolve_model_path(cfg: DictConfig) -> str:
    """Return a local .zip path, downloading from wandb if model_artifact is set."""
    artifact_ref = cfg.eval_rl.get("model_artifact", None)
    local_path = cfg.eval_rl.get("model_path", None)

    if artifact_ref:
        import wandb

        logger.info("downloading model artifact: %s", artifact_ref)
        run = wandb.init(
            project=cfg.wandb.project if "wandb" in cfg else "zetabench",
            job_type="eval",
            name=cfg.run_name,
        )
        artifact = run.use_artifact(str(artifact_ref), type="model")
        model_dir = artifact.download()
        path = str(Path(model_dir) / "model.zip")
        logger.info("artifact downloaded to %s", path)
        return path

    if local_path:
        logger.info("using local model path: %s", local_path)
        return str(local_path)

    raise ValueError(
        "Provide eval_rl.model_artifact (wandb registry ref) or "
        "eval_rl.model_path (local .zip path). Both are currently null."
    )


def _load_agent(agent_name: str, model_path: str) -> object:
    """Import and load the agent class for ``agent_name``."""
    if agent_name not in _AGENTS:
        raise ValueError(f"Unknown agent '{agent_name}'; expected one of {sorted(_AGENTS)}")
    module_path, class_name = _AGENTS[agent_name].rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    logger.info("loading %s from %s", class_name, model_path)
    return cls.load(model_path)


def _run_episode(
    env: RocketLandingEnv,
    agent: object,
    scaler: FixedObsScaler,
    seed: int,
    initial_fuel_kg: float,
    buffer: _TrajectoryBuffer | None = None,
) -> dict[str, float | str | int]:
    """Run one episode to termination/truncation; return per-episode metrics."""
    obs, _ = env.reset(seed=seed)

    ep_return = 0.0
    length = 0
    last_reason = "ongoing"

    while True:
        action = agent.predict(obs, deterministic=True)
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

    fps = int(cfg.eval_rl.get("render_fps", int(cfg.env.episode.control_hz)))
    scene_meta = {
        "pad_radius_m": float(cfg.env.touchdown.get("pad_radius_m", 30.0)),
        "oob_cylinder_radius_m": float(cfg.env.oob.cylinder_radius_m),
        "oob_ceiling_m": float(cfg.env.oob.ceiling_m),
        # Use touchdown speed threshold as the visual reference line
        "target_descent_mps": float(cfg.env.touchdown.get("velocity_mps", 2.0)),
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


@hydra.main(config_path="../configs", config_name="eval_rl", version_base=None)
def main(cfg: DictConfig) -> None:
    """Resolve model, build env, run n_episodes, dump CSV + JSON + stdout summary."""
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    agent_name = str(cfg.agent.name)
    logger.info("run_name=%s agent=%s results_dir=%s", cfg.run_name, agent_name, results_dir)
    logger.info(
        "n_episodes=%d seed=%d curriculum_progress=%.3f",
        int(cfg.eval_rl.n_episodes),
        int(cfg.seed),
        float(cfg.eval_rl.curriculum_progress),
    )

    model_path = _resolve_model_path(cfg)
    agent = _load_agent(agent_name, model_path)

    env = RocketLandingEnv(cfg)
    scaler = FixedObsScaler(cfg)
    initial_fuel_kg = float(cfg.env.dynamics.initial_fuel_kg)
    control_dt = 1.0 / float(cfg.env.episode.control_hz)

    render_enabled = bool(cfg.eval_rl.get("render", False))

    rng = np.random.default_rng(int(cfg.seed))
    rows: list[dict[str, float | str | int]] = []
    buffers: list[_TrajectoryBuffer | None] = []
    for ep_idx in range(int(cfg.eval_rl.n_episodes)):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        buffer = _TrajectoryBuffer(control_dt, scaler) if render_enabled else None
        row = _run_episode(env, agent, scaler, ep_seed, initial_fuel_kg, buffer=buffer)
        row["episode_idx"] = ep_idx
        rows.append(row)
        buffers.append(buffer)
        logger.info(
            "ep %03d/%03d seed=%d outcome=%-13s return=%9.2f len=%4d v_td=%5.2f m/s fuel=%6.1f kg",
            ep_idx + 1,
            int(cfg.eval_rl.n_episodes),
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
    # Gate check: warn explicitly if below the Phase 2 milestone target.
    if summary["success_rate"] < 0.70:
        logger.warning(
            "success_rate=%.1f%% is below the 70%% Phase 2 milestone target",
            100.0 * summary["success_rate"],
        )


if __name__ == "__main__":
    main()
