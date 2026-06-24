"""Hydra entrypoint: train an agent in nominal or adversarial mode.

CLI examples
------------
    python experiments/train.py                          # SAC, CPU, nominal
    python experiments/train.py seed=42 agent=ppo        # PPO
    python experiments/train.py compute=small_gpu        # switch to CUDA
    python experiments/train.py train_mode=adversarial   # Phase 3 (not yet)

Run-name convention: ``{agent}_{fidelity}_{train_mode}_{seed}``.

This module is an **entrypoint only** (per ``CONTRIBUTING.md`` §Module Dependency
Rules): it composes config, wires up wandb, and delegates the training loop and
checkpointing to the agent wrapper's ``learn``. No business logic lives here.

Adversarial training (alternating agent/adversary updates) lands in Phase 3 and
is intentionally gated off below.
"""
from __future__ import annotations

import hydra
import wandb
from dotenv import load_dotenv

load_dotenv()
from omegaconf import DictConfig, OmegaConf

from controllers.ppo_agent import PPOAgent
from controllers.sac_agent import SACAgent
from envs.rocket_landing_env import RocketLandingEnv
from utils.logging_config import get_logger
from utils.wandb_setup import ensure_project, register_resolvers

register_resolvers()

logger = get_logger(__name__)

_AGENTS = {"sac": SACAgent, "ppo": PPOAgent}


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> float | None:
    """Compose config, instantiate env + agent, run the nominal training loop.

    Returns the best evaluation mean reward (the objective the Optuna sweeper
    maximizes) for a learned agent, or ``None`` for non-learning paths (pid).
    """
    from pathlib import Path

    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    agent_name = str(cfg.agent.name)
    logger.info("run_name=%s agent=%s train_mode=%s seed=%d",
                cfg.run_name, agent_name, cfg.train_mode, int(cfg.seed))

    if str(cfg.train_mode) == "adversarial":
        raise NotImplementedError(
            "Adversarial training is a Phase 3 deliverable "
            "(adversary policy + alternating updates). Use train_mode=nominal."
        )

    if agent_name == "pid":
        # PID has no learnable parameters; evaluate it via evaluate_pid.py.
        logger.info("PID baseline has nothing to train; run experiments/evaluate_pid.py")
        return

    if agent_name not in _AGENTS:
        raise ValueError(f"Unknown agent '{agent_name}'; expected one of {sorted(_AGENTS)}")

    # Preflight: fail fast on a missing/invalid key and ensure the project exists
    # (no-op for offline runs). Runs before init so errors surface with context.
    wandb_entity = cfg.wandb.get("entity", None)
    ensure_project(cfg.wandb.project, entity=wandb_entity)

    wandb.init(
        entity=wandb_entity,
        project=cfg.wandb.project,
        name=cfg.run_name,
        mode=cfg.wandb.mode,
        tags=list(cfg.wandb.tags),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    try:
        env = RocketLandingEnv(cfg)
        agent = _AGENTS[agent_name](cfg)
        agent.learn(env, int(cfg.total_steps))
        logger.info("training complete; artefacts in %s", results_dir)
        _register_model(cfg, agent_name, results_dir)
        objective = _objective_from_evals(results_dir)
    finally:
        wandb.finish()

    # The Optuna sweeper (configs/hpo_*.yaml) maximizes this return value.
    logger.info("objective (best eval mean reward) = %.4f", objective)
    return objective


def _objective_from_evals(results_dir: "Path") -> float:
    """Best evaluation mean reward across all eval checkpoints in this run.

    Reads ``evaluations.npz`` written by SB3's ``EvalCallback`` (keys
    ``timesteps``/``results``/``ep_lengths``; ``results`` has shape
    ``(n_evals, n_eval_episodes)``) and returns ``results.mean(axis=1).max()`` —
    the highest mean reward any eval checkpoint achieved, i.e. the score of the
    ``best_model.zip`` the callback kept. This is the HPO objective.

    Raises if no eval ran (so misconfigured sweeps fail loudly rather than
    silently optimizing nothing): set ``eval_callback.every_n_steps`` below
    ``total_steps`` so at least one evaluation happens.
    """
    import numpy as np

    evals = results_dir / "evaluations.npz"
    if not evals.exists():
        raise RuntimeError(
            f"no evaluations.npz in {results_dir}; the EvalCallback never ran. "
            "Set eval_callback.every_n_steps < total_steps so at least one "
            "evaluation produces the HPO objective."
        )
    data = np.load(evals)
    return float(data["results"].mean(axis=1).max())


def _register_model(cfg: DictConfig, agent_name: str, results_dir: "Path") -> None:
    """Register the trained model in the W&B Model Registry, or fall back to disk.

    Online (``cfg.wandb.mode == "online"``): logs ``best_model.zip`` (or ``model.zip``)
    as a versioned artifact tagged with session-identifying metadata (run id/name/url,
    entity, project) and links it into the W&B Model Registry collection
    ``wandb-registry-model/zetabench-{agent}`` (created on first use) so it is formally
    registered, traceable to its run, and consumable by ``evaluate_rl.py``. Registration
    is **best-effort**: any failure is logged as a warning and never aborts training,
    since the trained model is always on disk regardless.

    Offline / no API key: makes **no** network call — the trained model simply stays
    as a zip in the repo under ``results/{run_name}/``; its path is logged so it can
    be loaded directly (``evaluate_rl.py eval_rl.model_path=...``). This preserves the
    "offline just works" contract (see :mod:`utils.wandb_setup`).
    """
    from pathlib import Path as _Path

    best = _Path(results_dir) / "best_model.zip"
    final = _Path(results_dir) / "model.zip"
    model_file = best if best.exists() else (final if final.exists() else None)
    if model_file is None:
        logger.warning("no model file found in %s; nothing to register", results_dir)
        return

    # wandb.run is non-None even offline, so gate the network path on the resolved mode.
    online = wandb.run is not None and str(cfg.wandb.mode) == "online"
    if not online:
        logger.info(
            "wandb offline; skipping Model Registry. Trained model available at %s",
            model_file,
        )
        return

    # Session-identifying attributes so the registered model is traceable back to
    # the exact W&B run that produced it (id/name/url/entity/project + run config).
    artifact = wandb.Artifact(
        name=f"zetabench-{agent_name}",
        type="model",
        metadata={
            "seed": int(cfg.seed),
            "total_steps": int(cfg.total_steps),
            "fidelity": str(cfg.env.dynamics.fidelity),
            "train_mode": str(cfg.train_mode),
            "source_file": model_file.name,
            "agent": agent_name,
            "run_id": wandb.run.id,
            "run_name": wandb.run.name,
            "run_path": "/".join(wandb.run.path),
            "run_url": wandb.run.url,
            "entity": wandb.run.entity,
            "project": wandb.run.project,
        },
    )
    artifact.add_file(str(model_file), name="model.zip")

    # Best-effort: a registry outage (missing/unreachable org registry, permissions,
    # SDK-version link semantics) must not fail an otherwise-successful training run.
    # The trained model already lives on disk under results/{run_name}/, so we log a
    # warning and point at it rather than raising. link_artifact auto-creates the
    # collection when it does not yet exist, so registration is register-if-missing.
    registry = str(cfg.wandb.get("registry", "wandb-registry-model")).rstrip("/")
    target = f"{registry}/zetabench-{agent_name}"
    try:
        logged = wandb.run.log_artifact(artifact)
        logged.wait()  # the version must be committed before it can be linked into the registry
        wandb.run.link_artifact(
            logged,
            target_path=target,
            aliases=["best", str(cfg.train_mode), str(cfg.env.dynamics.fidelity)],
        )
    except Exception as exc:  # noqa: BLE001 - registration is best-effort; never fatal
        wandb.run.summary["model_registered"] = False
        logger.warning(
            "W&B model registration failed (%r); model available on disk at %s",
            exc,
            model_file,
        )
        return

    wandb.run.summary["model_registered"] = True
    logger.info(
        "registered model in W&B Registry: %s (run %s, from %s)",
        target,
        "/".join(wandb.run.path),
        model_file.name,
    )


if __name__ == "__main__":
    main()
