"""Hydra entrypoint: end-to-end RL agent evaluation (SAC / PPO).

Loads a trained model — either from the wandb Model Registry or a local .zip
path — and runs it through :class:`envs.RocketLandingEnv` for
``cfg.eval_rl.n_episodes`` episodes. Writes per-episode rows and an aggregate
summary to disk and prints a human-readable summary to stdout.

CLI examples
------------
    # from wandb Model Registry (canonical); bare ref resolves entity/project
    # from wandb.* in configs/eval_rl.yaml
    python experiments/evaluate_rl.py agent=sac \\
      eval_rl.model_artifact="zetabench-sac:best"

    # fully-qualified registry ref
    python experiments/evaluate_rl.py agent=sac \\
      eval_rl.model_artifact="<entity>/wandb-registry-model/zetabench-sac:best"

    # local .zip path (offline / no wandb)
    python experiments/evaluate_rl.py agent=sac \\
      eval_rl.model_path=results/sac_moderate_nominal_42/best_model.zip

    # PPO, 200 episodes, full difficulty
    python experiments/evaluate_rl.py agent=ppo \\
      eval_rl.model_artifact="zetabench-ppo:best" \\
      eval_rl.n_episodes=200

    # Wind disturbance test (Phase 3 placeholder)
    python experiments/evaluate_rl.py agent=sac \\
      eval_rl.model_artifact="zetabench-sac:best" \\
      eval_rl.disturbance.wind_mps=5.0

Model resolution
----------------
``eval_rl.model_artifact`` takes precedence. When set, the ref is validated
(the docs placeholder ``entity/project/...`` is rejected), qualified with the
resolved ``wandb.entity``/``wandb.project``, downloaded via
``wandb.use_artifact``, and the local ``model.zip`` path is used. Falls back to
``eval_rl.model_path`` (local file) when artifact is null.

Outputs
-------
- Local checkpoints default to ``<model-dir>/eval_rl_p<progress>_seed<seed>/``.
- Artifact evals and explicit ``results_dir=...`` overrides write to
    ``results_dir``.
- ``episodes.csv`` — one row per episode.
- ``summary.json`` — aggregates across episodes.
- ``plots/`` + ``video/`` — optional renders.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import hydra
from dotenv import load_dotenv
import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from envs.rocket_landing_env import RocketLandingEnv
from utils.logging_config import get_logger
from utils.normalisation import FixedObsScaler
from utils.render import Trajectory, animate_side_view, plot_timeseries
from utils.wandb_setup import register_resolvers, resolve_wandb_mode

load_dotenv()
# Register ${zeta.wandb_mode:} before Hydra composes configs/eval_rl.yaml.
register_resolvers()

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


def _validate_artifact_ref(ref: str) -> str:
    """Validate the artifact ref's shape and reject the docs placeholder.

    Catches the example string ``entity/project/...:alias`` copied verbatim and
    missing-alias mistakes *before* any network call, so the failure is a clear,
    actionable message instead of a raw wandb ``CommError`` deep in
    ``use_artifact`` (``project 'project' not found under entity 'entity'``).
    """
    ref = ref.strip()
    if ":" not in ref:
        raise ValueError(
            f"Invalid eval_rl.model_artifact {ref!r}: missing ':<alias>' "
            "(e.g. 'zetabench-sac:best')."
        )
    parts = ref.split("/")
    if len(parts) > 3:
        raise ValueError(
            f"Invalid eval_rl.model_artifact {ref!r}: expected '<name>:<alias>', "
            "'<project>/<name>:<alias>' or '<entity>/<project>/<name>:<alias>'."
        )
    if len(parts) == 3 and parts[0] == "entity" and parts[1] == "project":
        raise ValueError(
            f"eval_rl.model_artifact={ref!r} is the documentation placeholder, "
            "not a real reference. Pass a real ref such as "
            "'<entity>/wandb-registry-model/zetabench-sac:best' or a bare "
            "'zetabench-sac:best' (entity/project filled from wandb.* in "
            "configs/eval_rl.yaml), or set eval_rl.model_path to a local .zip."
        )
    return ref


def _qualify_artifact_ref(ref: str, *, entity: str | None, project: str | None) -> str:
    """Prepend the resolved entity/project to a bare ``name:alias`` ref."""
    parts = ref.split("/")
    if len(parts) == 1:
        if not project:
            raise ValueError(
                f"eval_rl.model_artifact={ref!r} has no project segment and "
                "wandb.project is unset in configs/eval_rl.yaml; pass a "
                "fully-qualified ref or set wandb.project."
            )
        prefix = f"{entity}/{project}" if entity else project
        return f"{prefix}/{ref}"
    if len(parts) == 2 and entity:
        return f"{entity}/{ref}"
    return ref


def _resolve_model_path(cfg: DictConfig) -> str:
    """Return a local .zip path, downloading from wandb if model_artifact is set."""
    artifact_ref = cfg.eval_rl.get("model_artifact", None)
    local_path = cfg.eval_rl.get("model_path", None)

    if artifact_ref:
        import wandb

        wandb_cfg = cfg.get("wandb", {})
        project = wandb_cfg.get("project", "zeta-bench")
        entity = wandb_cfg.get("entity", None)
        mode = wandb_cfg.get("mode", None) or resolve_wandb_mode()

        # Validate up front so the docs placeholder / malformed refs fail fast
        # with a clear message instead of a raw wandb CommError.
        ref = _validate_artifact_ref(str(artifact_ref))

        logger.info("downloading model artifact: %s", ref)
        run = wandb.init(
            entity=entity,
            project=project,
            job_type="eval",
            name=cfg.run_name,
            mode=mode,
        )
        # Qualify a bare 'name:alias' with the resolved entity/project so the
        # lookup targets the right account rather than wandb's default guess.
        qualified = _qualify_artifact_ref(
            ref, entity=entity or run.entity, project=project
        )
        try:
            artifact = run.use_artifact(qualified, type="model")
            model_dir = artifact.download()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download wandb model artifact {qualified!r}: {exc}. "
                "Verify the ref exists and your WANDB_API_KEY has access to it, "
                "or set eval_rl.model_path to a local .zip."
            ) from exc
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


def _default_results_dir(cfg: DictConfig) -> Path:
    """Return the resolved default eval output directory for this config."""
    return Path("results") / str(cfg.run_name)


def _format_progress_for_path(progress: float) -> str:
    """Render curriculum progress as a compact path-safe token."""
    token = f"{float(progress):.3f}".rstrip("0").rstrip(".")
    token = token.replace("-", "neg").replace(".", "p")
    return f"p{token or '0'}"


def _resolve_results_dir(cfg: DictConfig, model_path: str) -> Path:
    """Choose where evaluation artefacts should be written.

    Local checkpoint evals should live next to the model they testify for,
    unless the caller explicitly overrides ``results_dir``. Artifact-based evals
    keep the configured generic results directory because their downloaded path
    is a wandb cache location, not a project run directory.
    """
    configured = Path(str(cfg.results_dir))
    if configured != _default_results_dir(cfg):
        return configured

    has_local_model = cfg.eval_rl.get("model_path", None)
    has_artifact_model = cfg.eval_rl.get("model_artifact", None)
    if has_local_model and not has_artifact_model:
        progress = _format_progress_for_path(
            float(cfg.eval_rl.curriculum_progress)
        )
        dirname = f"eval_rl_{progress}_seed{int(cfg.seed)}"
        return Path(model_path).parent / dirname

    return configured


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
    model_path = _resolve_model_path(cfg)
    results_dir = _resolve_results_dir(cfg, model_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    agent_name = str(cfg.agent.name)
    logger.info("run_name=%s agent=%s results_dir=%s", cfg.run_name, agent_name, results_dir)
    logger.info(
        "n_episodes=%d seed=%d curriculum_progress=%.3f",
        int(cfg.eval_rl.n_episodes),
        int(cfg.seed),
        float(cfg.eval_rl.curriculum_progress),
    )

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
