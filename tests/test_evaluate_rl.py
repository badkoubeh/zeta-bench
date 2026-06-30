"""Unit tests for :mod:`experiments.evaluate_rl` helpers."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from experiments.evaluate_rl import _resolve_results_dir


def _cfg(
    *,
    results_dir: str,
    model_path: str | None,
    model_artifact: str | None = None,
):
    return OmegaConf.create(
        {
            "seed": 42,
            "run_name": "sac_moderate_eval_rl_42",
            "results_dir": results_dir,
            "eval_rl": {
                "model_path": model_path,
                "model_artifact": model_artifact,
                "task_difficulty": 0.0,
            },
        }
    )


def test_local_checkpoint_eval_defaults_next_to_model() -> None:
    cfg = _cfg(
        results_dir="results/sac_moderate_eval_rl_42",
        model_path="results/sac_vertical_brake_oob_m4_42/best_model.zip",
    )

    assert _resolve_results_dir(
        cfg, "results/sac_vertical_brake_oob_m4_42/best_model.zip"
    ) == Path("results/sac_vertical_brake_oob_m4_42/eval_rl_td0_seed42")


def test_explicit_results_dir_override_wins() -> None:
    cfg = _cfg(
        results_dir="results/custom_eval_dir",
        model_path="results/sac_vertical_brake_oob_m4_42/best_model.zip",
    )

    assert _resolve_results_dir(
        cfg, "results/sac_vertical_brake_oob_m4_42/best_model.zip"
    ) == Path("results/custom_eval_dir")


def test_artifact_eval_keeps_configured_results_dir() -> None:
    cfg = _cfg(
        results_dir="results/sac_moderate_eval_rl_42",
        model_path=None,
        model_artifact="zetabench-sac:best",
    )

    assert _resolve_results_dir(cfg, "wandb-cache/model.zip") == Path(
        "results/sac_moderate_eval_rl_42"
    )
