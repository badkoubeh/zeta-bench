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
from omegaconf import DictConfig, OmegaConf

from controllers.ppo_agent import PPOAgent
from controllers.sac_agent import SACAgent
from envs.rocket_landing_env import RocketLandingEnv
from utils.logging_config import get_logger

logger = get_logger(__name__)

_AGENTS = {"sac": SACAgent, "ppo": PPOAgent}


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    """Compose config, instantiate env + agent, run the nominal training loop."""
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

    wandb.init(
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
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
