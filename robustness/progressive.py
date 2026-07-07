"""Stage sequencing for the progressive training profile.

The profile trains each agent through two chained regimes:

- **Stage A (naive)** — task-difficulty curriculum ramps 0→1 under nominal
  dynamics (domain randomization off).
- **Verification gate** — the stage's candidate checkpoints are evaluated at
  full task difficulty under nominal conditions
  (:mod:`robustness.verification`); Stage B starts only from a checkpoint that
  clears the configured success-rate threshold. A failed gate triggers a
  bounded number of Stage A extension runs before the agent's chain is
  aborted (recorded in the gate report; the next agent still runs).
- **Stage B (robust)** — resumes from the gate-selected checkpoint with the
  task difficulty pinned at full and domain randomization ramping disturbance
  severity 0→1 (``severity_anneal_steps``).

Anneal-step math: ``env.curriculum.anneal_steps`` and
``env.domain_randomization.severity_anneal_steps`` count **per-env** steps,
while ``total_steps`` counts global steps across ``compute.n_envs`` workers —
:func:`per_env_anneal_steps` converts a global-budget fraction into the
per-env anneal target. The per-env counters also reset whenever envs are
rebuilt, which is why extension and Stage B runs pin the curriculum
(``schedule=fixed``) instead of re-ramping.

Everything here is pure control flow over injected runners (no subprocesses,
no torch): the entrypoint ``experiments/train_profile.py`` supplies a
``StageRunner`` that executes ``experiments/train.py`` and a ``GateEvaluator``
that executes ``experiments/evaluate_rl.py`` and parses its ``summary.json``.

Import rule: may import from ``dynamics``/``envs``/``controllers``/``utils``;
never from ``experiments``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from omegaconf import DictConfig, OmegaConf

from robustness.verification import (
    CandidateResult,
    archive_candidates,
    candidate_from_summary,
    discover_candidates,
    gate_passed,
    select_candidate,
    write_gate_report,
)
from utils.logging_config import get_logger

logger = get_logger(__name__)

_KNOWN_AGENTS = ("sac", "ppo")

StageRunner = Callable[[Sequence[str]], int]
"""Runs one training stage (``experiments/train.py`` overrides) → exit code."""

GateEvaluator = Callable[[Sequence[str], Path], Mapping[str, float | int] | None]
"""Runs one gate evaluation (``experiments/evaluate_rl.py`` overrides, output
dir) → parsed ``summary.json``, or ``None`` when the evaluation failed."""


@dataclass(frozen=True)
class StageSpec:
    """One fully-resolved training stage: a run name plus its CLI overrides."""

    kind: str
    """``stage_a`` | ``stage_a_ext`` | ``stage_b``."""
    agent: str
    run_name: str
    results_dir: Path
    overrides: tuple[str, ...]
    """Complete override list for ``experiments/train.py`` (relative paths,
    resolved against the process working directory = repo root)."""


def per_env_anneal_steps(total_steps: int, n_envs: int, ramp_fraction: float) -> int:
    """Convert a global-budget ramp fraction into a per-env anneal target.

    The env's curriculum counter and the DR wrapper's severity counter advance
    once per **worker** step, while ``total_steps`` counts steps summed across
    all ``n_envs`` workers — e.g. 2M global steps on 4 envs is 500k per-env
    steps, so a 0.5 ramp fraction yields ``anneal_steps=250_000``.
    """
    return max(1, int(int(total_steps) * float(ramp_fraction) / max(1, int(n_envs))))


def validate_profile(profile: DictConfig, n_envs: int) -> None:
    """Fail fast on a profile config that would break or silently no-op a run.

    In particular, the eval cadence must sit below every stage budget:
    ``experiments/train.py`` derives its objective from ``evaluations.npz``
    and raises when the ``EvalCallback`` never fired.
    """
    agents = list(profile.agents)
    if not agents:
        raise ValueError("profile.agents is empty; list at least one of sac/ppo")
    for agent in agents:
        if str(agent) not in _KNOWN_AGENTS:
            raise ValueError(f"unknown agent {agent!r} in profile.agents; expected {_KNOWN_AGENTS}")

    for stage_key, fraction_key in (
        ("stage_a", "curriculum_ramp_fraction"),
        ("stage_b", "dr_ramp_fraction"),
    ):
        stage = profile[stage_key]
        if int(stage.total_steps) < 1:
            raise ValueError(f"profile.{stage_key}.total_steps must be >= 1")
        fraction = float(stage[fraction_key])
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"profile.{stage_key}.{fraction_key} must be in (0, 1]")
        if int(stage.eval_every_n_steps) >= int(stage.total_steps):
            raise ValueError(
                f"profile.{stage_key}.eval_every_n_steps must be < total_steps so the "
                "EvalCallback fires at least once (train.py raises otherwise)"
            )

    gate = profile.gate
    if not 0.0 <= float(gate.threshold) <= 1.0:
        raise ValueError("profile.gate.threshold must be in [0, 1]")
    if int(gate.n_episodes) < 1:
        raise ValueError("profile.gate.n_episodes must be >= 1")
    if not list(gate.candidates):
        raise ValueError("profile.gate.candidates is empty")
    if int(gate.max_extensions) < 0:
        raise ValueError("profile.gate.max_extensions must be >= 0")
    if int(gate.max_extensions) > 0 and int(gate.extension_steps) <= int(
        profile.stage_a.eval_every_n_steps
    ):
        raise ValueError(
            "profile.gate.extension_steps must exceed profile.stage_a.eval_every_n_steps "
            "so an extension run performs at least one model-selection eval"
        )
    if int(gate.seed_offset) in (0, 999):
        raise ValueError(
            "profile.gate.seed_offset must differ from 0 (training seed) and 999 "
            "(the EvalCallback's model-selection seed offset) to keep streams disjoint"
        )
    if int(n_envs) < 1:
        raise ValueError("compute.n_envs must be >= 1")


def _tags_override(profile: DictConfig, stage_tag: str, agent: str) -> str:
    tags = [str(t) for t in profile.wandb_tags] + [stage_tag, str(agent)]
    return f"wandb.tags=[{','.join(tags)}]"


def _common_overrides(
    profile: DictConfig,
    stage: DictConfig,
    agent: str,
    seed: int,
    fidelity: str,
    run_name: str,
    results_dir: Path,
) -> list[str]:
    return [
        f"agent={agent}",
        f"seed={int(seed)}",
        f"env.dynamics.fidelity={fidelity}",
        f"total_steps={int(stage.total_steps)}",
        f"run_name={run_name}",
        f"results_dir={results_dir.as_posix()}",
        f"eval_callback.every_n_steps={int(stage.eval_every_n_steps)}",
        f"eval_callback.n_eval_episodes={int(stage.n_eval_episodes)}",
        f"eval_callback.task_difficulty={float(stage.eval_task_difficulty)}",
    ]


def build_stage_a(
    profile: DictConfig, agent: str, seed: int, fidelity: str, n_envs: int
) -> StageSpec:
    """Naive stage: task-difficulty curriculum ramp, domain randomization off."""
    run_name = f"{agent}_{fidelity}_stageA_{int(seed)}"
    results_dir = Path("results") / run_name
    anneal = per_env_anneal_steps(
        int(profile.stage_a.total_steps), n_envs, float(profile.stage_a.curriculum_ramp_fraction)
    )
    overrides = _common_overrides(
        profile, profile.stage_a, agent, seed, fidelity, run_name, results_dir
    ) + [
        "env.curriculum.schedule=linear",
        f"env.curriculum.anneal_steps={anneal}",
        "env.domain_randomization.enabled=false",
        _tags_override(profile, "stage_a", agent),
        *[str(o) for o in profile.overrides],
    ]
    return StageSpec("stage_a", str(agent), run_name, results_dir, tuple(overrides))


def build_extension(
    profile: DictConfig, agent: str, seed: int, fidelity: str, attempt: int
) -> StageSpec:
    """Bounded Stage A retry: resume the final model, difficulty pinned at the gate's.

    Writes into the **same** results dir as Stage A (candidates and replay
    buffer stay in place for the next gate attempt) under a distinct run name.
    The curriculum is pinned fixed at ``gate.task_difficulty`` because the
    per-env ramp counter resets when the envs are rebuilt — the ramp already
    completed in Stage A.
    """
    stage_a_run = f"{agent}_{fidelity}_stageA_{int(seed)}"
    results_dir = Path("results") / stage_a_run
    run_name = f"{stage_a_run}_ext{int(attempt)}"
    ext_stage = OmegaConf.create(
        {
            "total_steps": int(profile.gate.extension_steps),
            "eval_every_n_steps": int(profile.stage_a.eval_every_n_steps),
            "n_eval_episodes": int(profile.stage_a.n_eval_episodes),
            "eval_task_difficulty": float(profile.stage_a.eval_task_difficulty),
        }
    )
    overrides = _common_overrides(
        profile, ext_stage, agent, seed, fidelity, run_name, results_dir
    ) + [
        f"resume_from={(results_dir / 'model.zip').as_posix()}",
        "env.curriculum.schedule=fixed",
        f"env.curriculum.task_difficulty={float(profile.gate.task_difficulty)}",
        "env.domain_randomization.enabled=false",
        _tags_override(profile, "stage_a_ext", agent),
        *[str(o) for o in profile.overrides],
    ]
    return StageSpec("stage_a_ext", str(agent), run_name, results_dir, tuple(overrides))


def build_stage_b(
    profile: DictConfig,
    agent: str,
    seed: int,
    fidelity: str,
    n_envs: int,
    resume_checkpoint: Path,
) -> StageSpec:
    """Robust stage: resume the gated checkpoint, ramp DR severity 0→1.

    Disturbance *ranges* stay in ``configs/env.yaml`` (the single source of
    magnitudes); this stage only enables the wrapper and sizes its ramp.
    """
    run_name = f"{agent}_{fidelity}_dr_{int(seed)}"
    results_dir = Path("results") / run_name
    severity_anneal = per_env_anneal_steps(
        int(profile.stage_b.total_steps), n_envs, float(profile.stage_b.dr_ramp_fraction)
    )
    overrides = _common_overrides(
        profile, profile.stage_b, agent, seed, fidelity, run_name, results_dir
    ) + [
        f"resume_from={Path(resume_checkpoint).as_posix()}",
        "env.curriculum.schedule=fixed",
        f"env.curriculum.task_difficulty={float(profile.stage_b.task_difficulty)}",
        "env.domain_randomization.enabled=true",
        f"env.domain_randomization.severity_anneal_steps={severity_anneal}",
        _tags_override(profile, "stage_b_dr", agent),
        *[str(o) for o in profile.overrides],
    ]
    return StageSpec("stage_b", str(agent), run_name, results_dir, tuple(overrides))


def build_gate_overrides(
    profile: DictConfig,
    agent: str,
    seed: int,
    fidelity: str,
    candidate: Path,
    out_dir: Path,
) -> tuple[str, ...]:
    """Override list for one ``experiments/evaluate_rl.py`` gate evaluation."""
    return (
        f"agent={agent}",
        f"seed={int(seed) + int(profile.gate.seed_offset)}",
        f"env.dynamics.fidelity={fidelity}",
        f"eval_rl.model_path={Path(candidate).as_posix()}",
        f"eval_rl.task_difficulty={float(profile.gate.task_difficulty)}",
        f"eval_rl.n_episodes={int(profile.gate.n_episodes)}",
        "eval_rl.render=false",
        f"results_dir={Path(out_dir).as_posix()}",
    )


def _evaluate_attempt(
    profile: DictConfig,
    agent: str,
    seed: int,
    fidelity: str,
    stage_a_dir: Path,
    attempt: int,
    gate_dir: Path,
    gate_evaluator: GateEvaluator,
) -> list[CandidateResult]:
    """Archive the stage's candidates and score each with the gate evaluator."""
    archive_dir = archive_candidates(stage_a_dir, attempt, list(profile.gate.candidates))
    results: list[CandidateResult] = []
    for candidate in discover_candidates(archive_dir, list(profile.gate.candidates)):
        out_dir = gate_dir / f"{agent}_attempt{attempt}_{candidate.stem}"
        overrides = build_gate_overrides(profile, agent, seed, fidelity, candidate, out_dir)
        summary = gate_evaluator(overrides, out_dir)
        if summary is None:
            logger.warning("gate evaluation failed for %s; skipping candidate", candidate)
            continue
        results.append(candidate_from_summary(candidate.name, candidate, summary))
    return results


def run_profile(
    cfg: DictConfig,
    runner: StageRunner,
    gate_evaluator: GateEvaluator,
    report_path: Path,
) -> dict[str, object]:
    """Run the full profile: per agent, Stage A → gate (→ extensions) → Stage B.

    A failed agent chain (crashed stage or failed gate) is recorded in the
    report and the next agent still runs — a one-shot overnight invocation
    never dies silently halfway. The report is rewritten after every step, so
    an interrupted run leaves a parseable record.
    """
    profile = cfg.profile
    seed = int(cfg.seed)
    fidelity = str(cfg.env.dynamics.fidelity)
    n_envs = int(cfg.compute.n_envs)
    validate_profile(profile, n_envs)

    gate_dir = Path(str(cfg.results_dir)) / "gate"
    agents_report: dict[str, dict[str, object]] = {}
    report: dict[str, object] = {
        "profile": str(profile.name),
        "seed": seed,
        "fidelity": fidelity,
        "n_envs": n_envs,
        "gate_threshold": float(profile.gate.threshold),
        "agents": agents_report,
    }

    def checkpoint_report() -> None:
        write_gate_report(report_path, report)

    for agent in [str(a) for a in profile.agents]:
        gate_attempts: list[dict[str, object]] = []
        extensions: list[dict[str, object]] = []
        entry: dict[str, object] = {
            "status": "running",
            "gate_attempts": gate_attempts,
            "extensions": extensions,
        }
        agents_report[agent] = entry
        checkpoint_report()

        stage_a = build_stage_a(profile, agent, seed, fidelity, n_envs)
        rc = runner(stage_a.overrides)
        entry["stage_a"] = {"run_name": stage_a.run_name, "exit_code": rc}
        if rc != 0:
            entry["status"] = "stage_crashed"
            checkpoint_report()
            logger.error("[%s] stage A exited %d; aborting this agent's chain", agent, rc)
            continue

        all_results: list[CandidateResult] = []
        attempt = 0
        passed = False
        while True:
            attempt_results = _evaluate_attempt(
                profile, agent, seed, fidelity, stage_a.results_dir, attempt, gate_dir,
                gate_evaluator,
            )
            attempt_record: dict[str, object] = {
                "attempt": attempt,
                "candidates": [c.as_report_entry() for c in attempt_results],
            }
            gate_attempts.append(attempt_record)
            if not attempt_results:
                entry["status"] = "gate_failed"
                attempt_record["error"] = "no scorable candidates"
                checkpoint_report()
                break
            all_results.extend(attempt_results)
            selected = select_candidate(attempt_results)
            attempt_record["selected"] = selected.as_report_entry()
            attempt_record["passed"] = gate_passed(selected, float(profile.gate.threshold))
            checkpoint_report()
            logger.info(
                "[%s] gate attempt %d: %s success_rate=%.1f%% (threshold %.1f%%)",
                agent, attempt, selected.name, 100.0 * selected.success_rate,
                100.0 * float(profile.gate.threshold),
            )
            if attempt_record["passed"]:
                passed = True
                break
            if attempt >= int(profile.gate.max_extensions):
                entry["status"] = "gate_failed"
                checkpoint_report()
                logger.error(
                    "[%s] gate failed after %d attempt(s); skipping stage B", agent, attempt + 1
                )
                break
            extension = build_extension(profile, agent, seed, fidelity, attempt + 1)
            rc = runner(extension.overrides)
            extensions.append({"run_name": extension.run_name, "exit_code": rc})
            if rc != 0:
                entry["status"] = "stage_crashed"
                checkpoint_report()
                logger.error("[%s] extension exited %d; aborting chain", agent, rc)
                break
            attempt += 1

        if not passed:
            continue

        best_overall = select_candidate(all_results)
        entry["selected_checkpoint"] = str(best_overall.path)
        checkpoint_report()

        stage_b = build_stage_b(profile, agent, seed, fidelity, n_envs, best_overall.path)
        rc = runner(stage_b.overrides)
        entry["stage_b"] = {"run_name": stage_b.run_name, "exit_code": rc}
        entry["status"] = "completed" if rc == 0 else "stage_crashed"
        checkpoint_report()

    return report
