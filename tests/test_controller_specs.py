"""Unit tests for named-controller resolution in the robustness matrix.

Covers the config seam that lets one matrix run compare multiple variants of
the same algorithm (e.g. ``sac_naive`` / ``sac_robust`` with ``type: sac``)
while staying backward compatible with the plain ``pid``/``sac``/``ppo``
entries. RL loading is faked — no torch needed.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from robustness.evaluation import ControllerSpec, resolve_controller_specs


@pytest.fixture(autouse=True)
def _clear_hydra():
    """Reset Hydra's global singleton around each test (compose hygiene)."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    yield
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


class TestResolveControllerSpecs:
    def test_backward_compatible_plain_names(self) -> None:
        cfg = OmegaConf.create(
            {
                "pid": {"enabled": True},
                "sac": {"enabled": True, "model_path": "results/sac/best_model.zip"},
                "ppo": {"enabled": False, "model_path": "results/ppo/best_model.zip"},
            }
        )
        specs = resolve_controller_specs(cfg)
        assert specs == [
            ControllerSpec("pid", "pid", None, True),
            ControllerSpec("sac", "sac", "results/sac/best_model.zip", True),
            ControllerSpec("ppo", "ppo", "results/ppo/best_model.zip", False),
        ]

    def test_named_variants_with_explicit_type(self) -> None:
        cfg = OmegaConf.create(
            {
                "sac_naive": {"enabled": True, "type": "sac", "model_path": "a.zip"},
                "sac_robust": {"enabled": True, "type": "sac", "model_path": "b.zip"},
            }
        )
        specs = resolve_controller_specs(cfg)
        assert [(s.name, s.kind, s.model_path) for s in specs] == [
            ("sac_naive", "sac", "a.zip"),
            ("sac_robust", "sac", "b.zip"),
        ]

    def test_config_order_preserved(self) -> None:
        cfg = OmegaConf.create(
            {
                "ppo": {"enabled": True},
                "pid": {"enabled": True},
                "sac": {"enabled": True},
            }
        )
        assert [s.name for s in resolve_controller_specs(cfg)] == ["ppo", "pid", "sac"]

    def test_unknown_name_without_type_raises(self) -> None:
        cfg = OmegaConf.create({"sac_robust": {"enabled": True, "model_path": "b.zip"}})
        with pytest.raises(ValueError, match="unknown kind 'sac_robust'"):
            resolve_controller_specs(cfg)

    def test_unknown_type_raises(self) -> None:
        cfg = OmegaConf.create({"agent": {"enabled": True, "type": "ddpg"}})
        with pytest.raises(ValueError, match="unknown kind 'ddpg'"):
            resolve_controller_specs(cfg)


class TestBuildControllers:
    def _cfg(self, config_name: str = "eval_robustness", overrides: list[str] | None = None):
        with initialize(config_path="../configs", version_base=None):
            return compose(config_name=config_name, overrides=overrides or [])

    def test_named_variants_loaded_and_missing_paths_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from experiments import evaluate_robustness

        loaded: list[tuple[str, str]] = []
        monkeypatch.setattr(
            evaluate_robustness,
            "_load_rl_agent",
            lambda kind, path: loaded.append((kind, path)) or f"fake-{kind}",
        )

        existing = tmp_path / "best_model.zip"
        existing.write_bytes(b"ckpt")
        cfg = self._cfg(
            overrides=[
                "eval_robustness.controllers.sac.enabled=false",
                "eval_robustness.controllers.ppo.enabled=false",
                "+eval_robustness.controllers.ppo_robust.enabled=true",
                "+eval_robustness.controllers.ppo_robust.type=ppo",
                f"+eval_robustness.controllers.ppo_robust.model_path={existing}",
                "+eval_robustness.controllers.sac_robust.enabled=true",
                "+eval_robustness.controllers.sac_robust.type=sac",
                "+eval_robustness.controllers.sac_robust.model_path=results/missing.zip",
            ]
        )
        controllers = evaluate_robustness._build_controllers(cfg)

        # pid built from config; ppo_robust loaded via its type; sac_robust
        # (enabled but missing checkpoint) warned and skipped.
        assert set(controllers) == {"pid", "ppo_robust"}
        assert controllers["ppo_robust"] == "fake-ppo"
        assert loaded == [("ppo", str(existing))]

    def test_profile_config_composes_with_expected_variants(self) -> None:
        cfg = self._cfg(config_name="eval_robustness_profile")
        specs = {s.name: s for s in resolve_controller_specs(cfg.eval_robustness.controllers)}

        assert not specs["sac"].enabled and not specs["ppo"].enabled
        enabled = [s for s in specs.values() if s.enabled]
        assert {s.name for s in enabled} == {
            "pid", "sac_naive", "sac_robust", "ppo_naive", "ppo_robust",
        }
        assert {s.kind for s in enabled if s.name != "pid"} == {"sac", "ppo"}
        # The matrix still pins initial conditions (fairness invariant).
        assert str(cfg.env.curriculum.schedule) == "fixed"
        assert float(cfg.env.curriculum.task_difficulty) == 1.0
