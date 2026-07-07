"""Hydra entrypoint: one-shot progressive training profile.

Chains, per agent (SAC, PPO): Stage A (task-difficulty curriculum, nominal
dynamics) → verification gate (success rate at full difficulty) → Stage B
(resume under domain randomization with a disturbance-severity ramp).

CLI examples
------------
    python experiments/train_profile.py                    # SAC + PPO, full budgets
    python experiments/train_profile.py profile=smoke      # minutes-scale wiring check
    python experiments/train_profile.py profile.agents=[sac] compute=cpu

This module is an **entrypoint only** (per ``CONTRIBUTING.md`` §Module
Dependency Rules): sequencing lives in :mod:`robustness.progressive`. Each
training stage runs as a ``python experiments/train.py <overrides>``
subprocess and each gate evaluation as ``python experiments/evaluate_rl.py
<overrides>`` — every stage stays an ordinary, individually reproducible run
(checkpointing, EvalCallback, wandb lifecycle), and a crash in one stage
cannot take down the orchestrator or the other agent's chain.

Outputs: per-stage artefacts under ``results/<stage_run_name>/`` as usual,
plus ``results/<run_name>/gate_report.json`` (per-candidate gate success
rates, selected checkpoints, per-agent status) and the gate evaluation runs
under ``results/<run_name>/gate/``.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import hydra
from dotenv import load_dotenv

load_dotenv()
from omegaconf import DictConfig

from robustness.progressive import run_profile
from utils.logging_config import get_logger

logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_script(script: str, overrides: Sequence[str]) -> int:
    """Run one entrypoint subprocess from the repo root; return its exit code."""
    cmd = [sys.executable, script, *overrides]
    logger.info("running: %s", " ".join(cmd))
    return subprocess.run(cmd, cwd=_REPO_ROOT, check=False).returncode


def _train_stage(overrides: Sequence[str]) -> int:
    return _run_script("experiments/train.py", overrides)


def _gate_evaluate(overrides: Sequence[str], out_dir: Path) -> Mapping[str, float | int] | None:
    """Run one gate evaluation and parse its ``summary.json`` (None on failure)."""
    rc = _run_script("experiments/evaluate_rl.py", overrides)
    summary_path = _REPO_ROOT / out_dir / "summary.json"
    if rc != 0 or not summary_path.exists():
        logger.warning("gate evaluation failed (exit=%d, summary=%s)", rc, summary_path)
        return None
    return json.loads(summary_path.read_text())


@hydra.main(config_path="../configs", config_name="train_profile", version_base=None)
def main(cfg: DictConfig) -> None:
    """Validate the profile, run the stage/gate chain, exit non-zero on failure."""
    from hydra.core.hydra_config import HydraConfig

    # Forward the compute group choice so each training subprocess composes the
    # same n_envs/device the anneal-step math here was derived from.
    compute_choice = str(HydraConfig.get().runtime.choices.get("compute", "cpu"))

    def runner(overrides: Sequence[str]) -> int:
        return _train_stage([*overrides, f"compute={compute_choice}"])

    report_path = Path(str(cfg.results_dir)) / "gate_report.json"
    report = run_profile(cfg, runner, _gate_evaluate, report_path)

    failed = []
    for agent, entry in report["agents"].items():  # type: ignore[union-attr]
        status = entry["status"]
        logger.info("[%s] final status: %s", agent, status)
        if status != "completed":
            failed.append(agent)
    logger.info("gate report: %s", report_path)
    if failed:
        logger.error("profile incomplete for: %s", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
