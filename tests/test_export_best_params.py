"""Tests for experiments/export_best_params.py.

Deterministic and training-free: they exercise the param routing, the tuned-config
rendering, and reading the sweeper's ``optimization_results.yaml`` (no DB, no
training, no Optuna study required).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from experiments.export_best_params import (
    read_best_params,
    render_tuned_config,
    resolve_results_file,
    route_best_params,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_route_splits_agent_from_compute() -> None:
    best = {
        "agent.learning_rate": 1.0e-4,
        "agent.gamma": 0.97,
        "agent.ent_coef": "auto",
        "compute.batch_size": 128,
        "compute.n_steps": 2048,
    }
    agent, compute = route_best_params(best)
    assert agent == {"learning_rate": 1.0e-4, "gamma": 0.97, "ent_coef": "auto"}
    assert compute == {"batch_size": 128, "n_steps": 2048}


def test_render_bakes_agent_and_excludes_compute() -> None:
    base = OmegaConf.create(
        {"agent": {"name": "sac", "learning_rate": 3.0e-4, "gamma": 0.99, "tau": 0.005}}
    )
    text = render_tuned_config(
        base,
        {"learning_rate": 1.0e-4, "gamma": 0.97},
        agent_name="sac",
        source="multirun/x/optimization_results.yaml",
        best_value=123.45,
        compute_overrides={"batch_size": 128},
    )

    # Exactly one package directive, on the first line.
    assert text.startswith("# @package _global_")
    assert text.count("# @package _global_") == 1

    loaded = OmegaConf.create(text)
    assert loaded.agent.learning_rate == 1.0e-4   # tuned value baked
    assert loaded.agent.gamma == 0.97             # tuned value baked
    assert loaded.agent.tau == 0.005              # untouched base value preserved
    assert "batch_size" not in loaded.agent       # compute.* not baked into agent

    # compute.* winner is reported in the header for manual reproduction.
    assert "compute.batch_size=128" in text


def test_render_handles_missing_best_value() -> None:
    base = OmegaConf.create({"agent": {"name": "sac", "gamma": 0.99}})
    text = render_tuned_config(
        base,
        {"gamma": 0.97},
        agent_name="sac",
        source="x",
        best_value=None,
        compute_overrides={},
    )
    assert "n/a" in text
    assert OmegaConf.create(text).agent.gamma == 0.97


def test_read_best_params_from_results_yaml(tmp_path: Path) -> None:
    results = tmp_path / "optimization_results.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "name": "optuna",
                "best_params": {"agent.learning_rate": 1.0e-4, "compute.batch_size": 256},
                "best_value": 42.0,
            }
        ),
        results,
    )
    params, value = read_best_params(results)
    assert params == {"agent.learning_rate": 1.0e-4, "compute.batch_size": 256}
    assert value == 42.0


def test_resolve_picks_latest_multirun(tmp_path: Path) -> None:
    older = tmp_path / "2026-06-24" / "10-00-00"
    newer = tmp_path / "2026-06-25" / "12-00-00"
    for d in (older, newer):
        d.mkdir(parents=True)
        (d / "optimization_results.yaml").write_text("name: optuna\nbest_params: {}\n")
    # Make `newer` unambiguously more recent regardless of creation order.
    import os
    import time

    os.utime(newer / "optimization_results.yaml", (time.time() + 10, time.time() + 10))

    resolved = resolve_results_file(None, tmp_path)
    assert resolved == newer / "optimization_results.yaml"


def test_resolve_accepts_explicit_dir_and_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    f = run_dir / "optimization_results.yaml"
    f.write_text("name: optuna\nbest_params: {}\n")
    assert resolve_results_file(run_dir, tmp_path) == f      # dir -> finds file inside
    assert resolve_results_file(f, tmp_path) == f            # file -> used directly


def test_base_agent_configs_are_loadable() -> None:
    # The exporter overlays onto these; guard that they exist and parse.
    for agent in ("sac", "ppo"):
        cfg = OmegaConf.load(_REPO_ROOT / "configs" / "agent" / f"{agent}.yaml")
        assert cfg.agent.name == agent
