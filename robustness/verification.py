"""Checkpoint verification gate used by the progressive training profile.

After a training stage completes, the gate evaluates the stage's candidate
checkpoints (``best_model.zip``, ``model.zip``) at full task difficulty under
nominal conditions and selects the strongest one. Training proceeds to the
next stage only when the selected candidate clears a configured success-rate
threshold (see :mod:`robustness.progressive` for the sequencing).

Candidates are copied into an immutable per-attempt archive before scoring:
a follow-up extension run re-creates SB3's ``EvalCallback`` with a fresh
``best_mean_reward = -inf``, whose first evaluation would otherwise overwrite
``best_model.zip`` even when it is worse than the checkpoint the gate scored.
The archive also carries ``replay_buffer.pkl`` when present so a SAC resume
from an archived checkpoint restores a warm buffer. Note the buffer belongs to
the *final* model of the stage; resuming ``best_model.zip`` with it is an
accepted approximation for an off-policy learner.

This module is pure bookkeeping (filesystem + selection arithmetic) — the
actual episode rollouts are delegated to ``experiments/evaluate_rl.py``, whose
``summary.json`` supplies the success rates consumed here.

Import rule: may import from ``dynamics``/``envs``/``controllers``/``utils``;
never from ``experiments``.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CandidateResult:
    """Gate evaluation outcome for one archived candidate checkpoint."""

    name: str
    """Candidate file name, e.g. ``best_model.zip``."""
    path: Path
    """Immutable archived path that was evaluated (safe to resume from later)."""
    success_rate: float
    return_mean: float
    n_episodes: int
    summary: Mapping[str, float | int]
    """Full ``summary.json`` payload from the gate evaluation run."""

    def as_report_entry(self) -> dict[str, object]:
        """JSON-serialisable record of this candidate for the gate report."""
        return {
            "name": self.name,
            "path": str(self.path),
            "success_rate": self.success_rate,
            "return_mean": self.return_mean,
            "n_episodes": self.n_episodes,
        }


def discover_candidates(directory: Path, names: Sequence[str]) -> list[Path]:
    """Return the candidate checkpoints that exist in ``directory``.

    ``names`` is the configured priority order (``gate.candidates``); the
    returned list preserves it, which is what gives earlier names precedence
    on exact ties in :func:`select_candidate`.
    """
    return [directory / name for name in names if (directory / name).exists()]


def archive_candidates(
    results_dir: Path,
    attempt: int,
    names: Sequence[str],
    extras: Sequence[str] = ("replay_buffer.pkl",),
) -> Path:
    """Copy candidate checkpoints into an immutable per-attempt archive.

    Copies each existing candidate (plus ``extras`` such as the SAC replay
    buffer) from ``results_dir`` to ``results_dir/gate_archive/attempt{k}/``
    so that later training runs cannot mutate what the gate scored. Missing
    files are skipped silently — e.g. PPO has no replay buffer. Returns the
    archive directory (created even when empty, so callers can uniformly
    discover candidates from it).
    """
    archive_dir = Path(results_dir) / "gate_archive" / f"attempt{int(attempt)}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for name in tuple(names) + tuple(extras):
        src = Path(results_dir) / name
        if src.exists():
            shutil.copy2(src, archive_dir / name)
            logger.info("archived %s -> %s", src, archive_dir / name)
    return archive_dir


def select_candidate(results: Sequence[CandidateResult]) -> CandidateResult:
    """Pick the strongest candidate: argmax success rate.

    Ties are broken by mean return, then by input order (the configured
    ``gate.candidates`` priority, ``best_model.zip`` first). Raises
    ``ValueError`` on an empty sequence so a stage that produced no scorable
    checkpoint fails loudly.
    """
    if not results:
        raise ValueError("select_candidate needs at least one CandidateResult")
    best = results[0]
    for cand in results[1:]:
        if cand.success_rate > best.success_rate or (
            cand.success_rate == best.success_rate and cand.return_mean > best.return_mean
        ):
            best = cand
    return best


def gate_passed(selected: CandidateResult, threshold: float) -> bool:
    """True when the selected candidate meets the success-rate threshold."""
    return selected.success_rate >= float(threshold)


def candidate_from_summary(
    name: str, path: Path, summary: Mapping[str, float | int]
) -> CandidateResult:
    """Build a :class:`CandidateResult` from an ``evaluate_rl`` summary payload."""
    return CandidateResult(
        name=name,
        path=path,
        success_rate=float(summary["success_rate"]),
        return_mean=float(summary["return_mean"]),
        n_episodes=int(summary["n_episodes"]),
        summary=dict(summary),
    )


def write_gate_report(path: Path, report: Mapping[str, object]) -> None:
    """Atomically write the gate report as JSON (tmp file + rename).

    The report is rewritten after every stage/gate step so an interrupted
    profile run still leaves a parseable record of how far it got.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=False))
    tmp.replace(out)
