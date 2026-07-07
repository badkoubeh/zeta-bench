"""Unit tests for the progressive-profile stage sequencing.

Everything runs against fake runners/evaluators (no torch, no subprocesses):
the fake runner materialises checkpoint files exactly where a real
``experiments/train.py`` run would, and the fake gate evaluator fabricates
``summary.json`` payloads. Filesystem writes go to ``tmp_path`` via
``monkeypatch.chdir`` since the stage specs use repo-root-relative paths.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import pytest
from omegaconf import DictConfig, OmegaConf

from robustness.progressive import (
    build_extension,
    build_gate_overrides,
    build_stage_a,
    build_stage_b,
    per_env_anneal_steps,
    run_profile,
    validate_profile,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _cfg(**profile_updates: object) -> DictConfig:
    """Small synthetic profile config mirroring configs/profile/*.yaml."""
    cfg = OmegaConf.create(
        {
            "profile": {
                "name": "test",
                "agents": ["sac"],
                "stage_a": {
                    "total_steps": 1000,
                    "curriculum_ramp_fraction": 0.5,
                    "eval_every_n_steps": 100,
                    "n_eval_episodes": 2,
                    "eval_task_difficulty": 1.0,
                },
                "gate": {
                    "threshold": 0.9,
                    "n_episodes": 4,
                    "task_difficulty": 1.0,
                    "seed_offset": 10_000,
                    "candidates": ["best_model.zip", "model.zip"],
                    "max_extensions": 1,
                    "extension_steps": 500,
                },
                "stage_b": {
                    "total_steps": 800,
                    "task_difficulty": 1.0,
                    "dr_ramp_fraction": 0.8,
                    "eval_every_n_steps": 100,
                    "n_eval_episodes": 2,
                    "eval_task_difficulty": 1.0,
                },
                "wandb_tags": ["t"],
                "overrides": [],
            },
            "seed": 42,
            "results_dir": "results/profile_test_moderate_42",
            "env": {"dynamics": {"fidelity": "moderate"}},
            "compute": {"n_envs": 4},
        }
    )
    for key, value in profile_updates.items():
        OmegaConf.update(cfg, f"profile.{key}", value)
    return cfg


def _get_override(overrides: Sequence[str], key: str) -> str:
    matches = [o.split("=", 1)[1] for o in overrides if o.startswith(f"{key}=")]
    assert len(matches) == 1, f"{key} not found exactly once in {overrides}"
    return matches[0]


class _FakeRunner:
    """Records stage invocations and materialises the checkpoints a run leaves."""

    def __init__(self, exit_codes: Sequence[int] = ()) -> None:
        self.calls: list[tuple[str, ...]] = []
        self._exit_codes = list(exit_codes)

    def __call__(self, overrides: Sequence[str]) -> int:
        self.calls.append(tuple(overrides))
        rc = self._exit_codes.pop(0) if self._exit_codes else 0
        if rc == 0:
            results_dir = Path(_get_override(overrides, "results_dir"))
            results_dir.mkdir(parents=True, exist_ok=True)
            (results_dir / "best_model.zip").write_bytes(b"best")
            (results_dir / "model.zip").write_bytes(b"final")
            (results_dir / "replay_buffer.pkl").write_bytes(b"buffer")
        return rc


class _FakeGate:
    """Returns scripted success rates (None simulates a failed evaluation)."""

    def __init__(self, rates: Sequence[float | None]) -> None:
        self.calls: list[tuple[tuple[str, ...], Path]] = []
        self._rates = list(rates)

    def __call__(
        self, overrides: Sequence[str], out_dir: Path
    ) -> Mapping[str, float | int] | None:
        self.calls.append((tuple(overrides), Path(out_dir)))
        rate = self._rates.pop(0)
        if rate is None:
            return None
        return {"success_rate": rate, "return_mean": 100.0 * rate, "n_episodes": 4}


class TestPerEnvAnnealSteps:
    def test_converts_global_budget_fraction(self) -> None:
        assert per_env_anneal_steps(2_000_000, 4, 0.5) == 250_000
        assert per_env_anneal_steps(1_000_000, 4, 0.8) == 200_000

    def test_single_env_passthrough(self) -> None:
        assert per_env_anneal_steps(1000, 1, 1.0) == 1000

    def test_floor_of_one(self) -> None:
        assert per_env_anneal_steps(1, 4, 0.1) == 1


class TestValidateProfile:
    def test_shipped_profiles_are_valid(self) -> None:
        for name in ("progressive", "smoke"):
            profile = OmegaConf.load(_REPO_ROOT / "configs" / "profile" / f"{name}.yaml")
            validate_profile(profile, n_envs=4)

    @pytest.mark.parametrize(
        ("update", "match"),
        [
            ({"agents": []}, "agents is empty"),
            ({"agents": ["ddpg"]}, "unknown agent"),
            ({"stage_a.total_steps": 0}, "total_steps"),
            ({"stage_a.curriculum_ramp_fraction": 0.0}, "curriculum_ramp_fraction"),
            ({"stage_b.dr_ramp_fraction": 1.2}, "dr_ramp_fraction"),
            ({"stage_a.eval_every_n_steps": 1000}, "EvalCallback fires"),
            ({"gate.threshold": 1.5}, "threshold"),
            ({"gate.n_episodes": 0}, "n_episodes"),
            ({"gate.candidates": []}, "candidates is empty"),
            ({"gate.max_extensions": -1}, "max_extensions"),
            ({"gate.extension_steps": 100}, "extension_steps"),
            ({"gate.seed_offset": 999}, "seed_offset"),
        ],
    )
    def test_rejects_broken_profiles(self, update: dict[str, object], match: str) -> None:
        cfg = _cfg(**update)
        with pytest.raises(ValueError, match=match):
            validate_profile(cfg.profile, n_envs=4)

    def test_rejects_bad_n_envs(self) -> None:
        with pytest.raises(ValueError, match="n_envs"):
            validate_profile(_cfg().profile, n_envs=0)


class TestStageSpecs:
    def test_stage_a_overrides(self) -> None:
        spec = build_stage_a(_cfg().profile, "sac", 42, "moderate", 4)
        assert spec.kind == "stage_a"
        assert spec.run_name == "sac_moderate_stageA_42"
        assert spec.results_dir == Path("results/sac_moderate_stageA_42")
        assert spec.overrides == (
            "agent=sac",
            "seed=42",
            "env.dynamics.fidelity=moderate",
            "total_steps=1000",
            "run_name=sac_moderate_stageA_42",
            "results_dir=results/sac_moderate_stageA_42",
            "eval_callback.every_n_steps=100",
            "eval_callback.n_eval_episodes=2",
            "eval_callback.task_difficulty=1.0",
            "env.curriculum.schedule=linear",
            "env.curriculum.anneal_steps=125",  # per_env(1000, 4, 0.5)
            "env.domain_randomization.enabled=false",
            "wandb.tags=[t,stage_a,sac]",
        )

    def test_extension_pins_difficulty_and_reuses_stage_a_dir(self) -> None:
        spec = build_extension(_cfg().profile, "sac", 42, "moderate", attempt=1)
        assert spec.kind == "stage_a_ext"
        assert spec.run_name == "sac_moderate_stageA_42_ext1"
        assert spec.results_dir == Path("results/sac_moderate_stageA_42")
        assert _get_override(spec.overrides, "total_steps") == "500"
        assert (
            _get_override(spec.overrides, "resume_from")
            == "results/sac_moderate_stageA_42/model.zip"
        )
        assert _get_override(spec.overrides, "env.curriculum.schedule") == "fixed"
        assert _get_override(spec.overrides, "env.curriculum.task_difficulty") == "1.0"
        assert _get_override(spec.overrides, "env.domain_randomization.enabled") == "false"
        assert _get_override(spec.overrides, "wandb.tags") == "[t,stage_a_ext,sac]"

    def test_stage_b_enables_dr_with_severity_ramp(self) -> None:
        checkpoint = Path("results/sac_moderate_stageA_42/gate_archive/attempt0/best_model.zip")
        spec = build_stage_b(_cfg().profile, "sac", 42, "moderate", 4, checkpoint)
        assert spec.kind == "stage_b"
        assert spec.run_name == "sac_moderate_dr_42"
        assert spec.results_dir == Path("results/sac_moderate_dr_42")
        assert _get_override(spec.overrides, "resume_from") == checkpoint.as_posix()
        assert _get_override(spec.overrides, "env.curriculum.schedule") == "fixed"
        assert _get_override(spec.overrides, "env.curriculum.task_difficulty") == "1.0"
        assert _get_override(spec.overrides, "env.domain_randomization.enabled") == "true"
        # per_env(800, 4, 0.8) = 160
        assert (
            _get_override(spec.overrides, "env.domain_randomization.severity_anneal_steps")
            == "160"
        )
        assert _get_override(spec.overrides, "wandb.tags") == "[t,stage_b_dr,sac]"

    def test_profile_overrides_appended_to_every_stage(self) -> None:
        profile = _cfg(overrides=["env.episode.max_steps=300"]).profile
        for spec in (
            build_stage_a(profile, "sac", 42, "moderate", 4),
            build_extension(profile, "sac", 42, "moderate", 1),
            build_stage_b(profile, "sac", 42, "moderate", 4, Path("ckpt.zip")),
        ):
            assert spec.overrides[-1] == "env.episode.max_steps=300"

    def test_gate_overrides(self) -> None:
        overrides = build_gate_overrides(
            _cfg().profile,
            "sac",
            42,
            "moderate",
            Path("results/sac_moderate_stageA_42/gate_archive/attempt0/best_model.zip"),
            Path("results/profile_test_moderate_42/gate/sac_attempt0_best_model"),
        )
        assert overrides == (
            "agent=sac",
            "seed=10042",
            "env.dynamics.fidelity=moderate",
            "eval_rl.model_path=results/sac_moderate_stageA_42/gate_archive/attempt0/"
            "best_model.zip",
            "eval_rl.task_difficulty=1.0",
            "eval_rl.n_episodes=4",
            "eval_rl.render=false",
            "results_dir=results/profile_test_moderate_42/gate/sac_attempt0_best_model",
        )


class TestRunProfile:
    def test_pass_first_try_runs_stage_b_from_archived_checkpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        runner = _FakeRunner()
        gate = _FakeGate([0.95, 0.92])  # best_model.zip, model.zip
        report = run_profile(_cfg(), runner, gate, tmp_path / "gate_report.json")

        entry = report["agents"]["sac"]
        assert entry["status"] == "completed"
        assert entry["gate_attempts"][0]["passed"] is True
        assert entry["selected_checkpoint"] == str(
            Path("results/sac_moderate_stageA_42/gate_archive/attempt0/best_model.zip")
        )
        # Two training runs: stage A then stage B resuming the archived checkpoint.
        assert len(runner.calls) == 2
        assert _get_override(runner.calls[1], "resume_from").endswith(
            "gate_archive/attempt0/best_model.zip"
        )
        # Both candidates were archived (with the replay buffer) before scoring.
        archive = Path("results/sac_moderate_stageA_42/gate_archive/attempt0")
        assert sorted(p.name for p in archive.iterdir()) == [
            "best_model.zip", "model.zip", "replay_buffer.pkl",
        ]

    def test_fail_then_extension_then_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        runner = _FakeRunner()
        gate = _FakeGate([0.5, 0.4, 0.95, 0.9])  # attempt0 fails, attempt1 passes
        report = run_profile(_cfg(), runner, gate, tmp_path / "gate_report.json")

        entry = report["agents"]["sac"]
        assert entry["status"] == "completed"
        assert [a["passed"] for a in entry["gate_attempts"]] == [False, True]
        assert entry["extensions"] == [
            {"run_name": "sac_moderate_stageA_42_ext1", "exit_code": 0}
        ]
        # Selection spans all attempts; the passing attempt-1 candidate wins here.
        assert "attempt1" in entry["selected_checkpoint"]
        # stage A, extension, stage B.
        assert len(runner.calls) == 3
        assert _get_override(runner.calls[1], "resume_from").endswith("model.zip")

    def test_gate_failure_skips_stage_b_but_next_agent_completes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cfg = _cfg(agents=["sac", "ppo"], **{"gate.max_extensions": 0})
        runner = _FakeRunner()
        gate = _FakeGate([0.5, 0.4, 0.95, 0.9])  # sac fails, ppo passes
        report = run_profile(cfg, runner, gate, tmp_path / "gate_report.json")

        assert report["agents"]["sac"]["status"] == "gate_failed"
        assert "selected_checkpoint" not in report["agents"]["sac"]
        assert report["agents"]["ppo"]["status"] == "completed"
        # sac: stage A only; ppo: stage A + stage B.
        assert len(runner.calls) == 3
        assert _get_override(runner.calls[0], "agent") == "sac"
        assert _get_override(runner.calls[1], "agent") == "ppo"

    def test_stage_a_crash_recorded_and_next_agent_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cfg = _cfg(agents=["sac", "ppo"])
        runner = _FakeRunner(exit_codes=[3])  # sac stage A crashes; later runs succeed
        gate = _FakeGate([0.95, 0.9])  # only ppo reaches the gate
        report = run_profile(cfg, runner, gate, tmp_path / "gate_report.json")

        assert report["agents"]["sac"]["status"] == "stage_crashed"
        assert report["agents"]["sac"]["stage_a"]["exit_code"] == 3
        assert report["agents"]["sac"]["gate_attempts"] == []
        assert report["agents"]["ppo"]["status"] == "completed"

    def test_extension_crash_aborts_chain(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        runner = _FakeRunner(exit_codes=[0, 2])  # stage A ok, extension crashes
        gate = _FakeGate([0.5, 0.4])
        report = run_profile(_cfg(), runner, gate, tmp_path / "gate_report.json")

        entry = report["agents"]["sac"]
        assert entry["status"] == "stage_crashed"
        assert entry["extensions"][0]["exit_code"] == 2
        assert len(runner.calls) == 2  # no stage B

    def test_all_evaluations_failing_is_gate_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        runner = _FakeRunner()
        gate = _FakeGate([None, None])
        report = run_profile(_cfg(), runner, gate, tmp_path / "gate_report.json")

        entry = report["agents"]["sac"]
        assert entry["status"] == "gate_failed"
        assert entry["gate_attempts"][0]["error"] == "no scorable candidates"
        assert len(runner.calls) == 1

    def test_report_written_incrementally(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        report_path = tmp_path / "gate_report.json"

        class _InterruptingGate(_FakeGate):
            def __call__(self, overrides: Sequence[str], out_dir: Path) -> None:
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            run_profile(_cfg(), _FakeRunner(), _InterruptingGate([]), report_path)

        # The interrupted run still left a parseable record of how far it got.
        on_disk = json.loads(report_path.read_text())
        assert on_disk["agents"]["sac"]["status"] == "running"

    def test_final_report_matches_disk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        report_path = tmp_path / "gate_report.json"
        report = run_profile(_cfg(), _FakeRunner(), _FakeGate([0.95, 0.9]), report_path)
        assert json.loads(report_path.read_text()) == report
