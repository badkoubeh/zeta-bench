"""Hydra entrypoint: run the graduated robustness disturbance matrix.

Evaluates every configured controller (PID, SAC, PPO) across the same
fixed-seed disturbance grid so results are reproducible and cross-comparable,
then writes the matrix CSV and the signature per-controller heatmap. The grid
orchestration + rollout live in :mod:`robustness.evaluation`; this file only
builds controllers and wires outputs.

CLI
---
::

    # all controllers (RL checkpoints must exist under results/)
    python experiments/evaluate_robustness.py

    # PID only (no trained models needed)
    python experiments/evaluate_robustness.py \\
        eval_robustness.controllers.sac.enabled=false \\
        eval_robustness.controllers.ppo.enabled=false

    # point RL controllers at specific checkpoints
    python experiments/evaluate_robustness.py \\
        eval_robustness.controllers.sac.model_path=results/sac_moderate_nominal_42/best_model.zip

Outputs
-------
- ``results/robustness_matrix.csv`` — long-format table, one row per
  (controller × disturbance cell) with success rate and secondary metrics.
- ``results/robustness_heatmap.png`` — one panel per controller.
- optional wandb ``Table`` when ``eval.outputs.log_wandb_table`` is true.

The adversarial / worst-case mode is a *separate*, secondary path and is never
merged into this comparable matrix.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from controllers.pid_baseline import PIDController
from robustness.evaluation import (
    MATRIX_COLUMNS,
    run_matrix,
    write_matrix_csv,
)
from robustness.heatmap import plot_robustness_heatmap
from utils.logging_config import get_logger
from utils.wandb_setup import register_resolvers

# Register the ${zeta.wandb_mode:} resolver before Hydra composes the config.
register_resolvers()

logger = get_logger(__name__)

# RL controller classes, loaded lazily from disk (SB3/torch only imported when
# an RL controller is actually enabled and its checkpoint exists).
_RL_AGENTS: dict[str, str] = {
    "sac": "controllers.sac_agent.SACAgent",
    "ppo": "controllers.ppo_agent.PPOAgent",
}


def _load_rl_agent(name: str, model_path: str) -> object:
    """Import the SB3 agent wrapper for ``name`` and load its checkpoint."""
    module_path, class_name = _RL_AGENTS[name].rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)
    logger.info("loading %s controller from %s", name, model_path)
    return cls.load(model_path)


def _build_controllers(cfg: DictConfig) -> dict[str, object]:
    """Assemble the controller set to compare on identical conditions.

    PID is built from config (no checkpoint). Each RL controller is loaded only
    when enabled *and* its ``model_path`` exists — an enabled-but-missing model
    is warned about and skipped so the matrix still runs with what is available
    (e.g. PID-only in CI). Never tune per controller: every controller here is
    scored on the same cells and seeds.
    """
    spec = cfg.eval_robustness.controllers
    controllers: dict[str, object] = {}

    if bool(spec.pid.enabled):
        controllers["pid"] = PIDController(cfg)

    for name in ("sac", "ppo"):
        cspec = spec[name]
        if not bool(cspec.enabled):
            continue
        model_path = OmegaConf.select(cspec, "model_path", default=None)
        if model_path and Path(str(model_path)).exists():
            controllers[name] = _load_rl_agent(name, str(model_path))
        else:
            logger.warning(
                "controller %s enabled but model_path %r not found — skipping",
                name,
                model_path,
            )
    return controllers


def _log_wandb_table(cfg: DictConfig, rows: list[dict[str, object]]) -> None:
    """Best-effort log of the matrix as a wandb Table (never blocks the run)."""
    try:
        import wandb

        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.run_name,
            mode=str(cfg.wandb.mode),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        table = wandb.Table(
            columns=list(MATRIX_COLUMNS),
            data=[[row.get(c) for c in MATRIX_COLUMNS] for row in rows],
        )
        run.log({"robustness_matrix": table})
        run.finish()
        logger.info("logged robustness_matrix wandb table (%d rows)", len(rows))
    except Exception as exc:  # noqa: BLE001 - logging must not fail the eval
        logger.warning("skipping wandb table log: %s", exc)


@hydra.main(config_path="../configs", config_name="eval_robustness", version_base=None)
def main(cfg: DictConfig) -> None:
    """Build controllers, sweep the disturbance matrix, write CSV + heatmap."""
    controllers = _build_controllers(cfg)
    if not controllers:
        raise RuntimeError(
            "no controllers available to evaluate — enable at least one and "
            "ensure any RL model_path exists"
        )
    logger.info("evaluating controllers: %s", ", ".join(sorted(controllers)))

    rows = run_matrix(cfg, controllers)

    write_matrix_csv(rows, cfg.eval.outputs.csv_path)
    episodes_per_cell = int(cfg.eval.seeds) * int(cfg.eval.episodes_per_seed)
    plot_robustness_heatmap(
        rows, cfg.eval.outputs.heatmap_path, episodes_per_cell=episodes_per_cell
    )

    if bool(cfg.eval.outputs.log_wandb_table):
        _log_wandb_table(cfg, rows)


if __name__ == "__main__":
    main()
