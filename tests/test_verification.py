"""Unit tests for the checkpoint verification gate primitives.

Pure filesystem + selection arithmetic — no torch, no env rollouts (the gate's
episode rollouts are delegated to ``experiments/evaluate_rl.py`` and are out of
scope here).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from robustness.verification import (
    CandidateResult,
    archive_candidates,
    candidate_from_summary,
    discover_candidates,
    gate_passed,
    select_candidate,
    write_gate_report,
)

_CANDIDATES = ("best_model.zip", "model.zip")


def _result(name: str, success_rate: float, return_mean: float = 0.0) -> CandidateResult:
    return CandidateResult(
        name=name,
        path=Path("archive") / name,
        success_rate=success_rate,
        return_mean=return_mean,
        n_episodes=4,
        summary={},
    )


class TestDiscoverCandidates:
    def test_returns_existing_in_priority_order(self, tmp_path: Path) -> None:
        (tmp_path / "model.zip").write_bytes(b"final")
        (tmp_path / "best_model.zip").write_bytes(b"best")
        found = discover_candidates(tmp_path, _CANDIDATES)
        assert [p.name for p in found] == ["best_model.zip", "model.zip"]

    def test_skips_missing(self, tmp_path: Path) -> None:
        (tmp_path / "model.zip").write_bytes(b"final")
        found = discover_candidates(tmp_path, _CANDIDATES)
        assert [p.name for p in found] == ["model.zip"]

    def test_empty_dir(self, tmp_path: Path) -> None:
        assert discover_candidates(tmp_path, _CANDIDATES) == []


class TestArchiveCandidates:
    def test_copies_candidates_and_replay_buffer(self, tmp_path: Path) -> None:
        (tmp_path / "best_model.zip").write_bytes(b"best")
        (tmp_path / "model.zip").write_bytes(b"final")
        (tmp_path / "replay_buffer.pkl").write_bytes(b"buffer")

        archive = archive_candidates(tmp_path, 0, _CANDIDATES)

        assert archive == tmp_path / "gate_archive" / "attempt0"
        assert (archive / "best_model.zip").read_bytes() == b"best"
        assert (archive / "model.zip").read_bytes() == b"final"
        assert (archive / "replay_buffer.pkl").read_bytes() == b"buffer"

    def test_archive_is_immutable_against_later_overwrites(self, tmp_path: Path) -> None:
        (tmp_path / "best_model.zip").write_bytes(b"good")
        archive = archive_candidates(tmp_path, 0, _CANDIDATES)
        # A later extension run overwrites the live checkpoint...
        (tmp_path / "best_model.zip").write_bytes(b"worse")
        # ...but the archived copy the gate scored is untouched.
        assert (archive / "best_model.zip").read_bytes() == b"good"

    def test_skips_missing_files(self, tmp_path: Path) -> None:
        (tmp_path / "model.zip").write_bytes(b"final")  # no best_model, no buffer (PPO)
        archive = archive_candidates(tmp_path, 1, _CANDIDATES)
        assert archive == tmp_path / "gate_archive" / "attempt1"
        assert sorted(p.name for p in archive.iterdir()) == ["model.zip"]

    def test_attempts_get_distinct_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "model.zip").write_bytes(b"v0")
        first = archive_candidates(tmp_path, 0, _CANDIDATES)
        (tmp_path / "model.zip").write_bytes(b"v1")
        second = archive_candidates(tmp_path, 1, _CANDIDATES)
        assert (first / "model.zip").read_bytes() == b"v0"
        assert (second / "model.zip").read_bytes() == b"v1"


class TestSelectCandidate:
    def test_argmax_success_rate(self) -> None:
        best = select_candidate([_result("best_model.zip", 0.7), _result("model.zip", 0.9)])
        assert best.name == "model.zip"

    def test_tie_broken_by_return_mean(self) -> None:
        best = select_candidate(
            [_result("best_model.zip", 0.8, return_mean=10.0),
             _result("model.zip", 0.8, return_mean=20.0)]
        )
        assert best.name == "model.zip"

    def test_full_tie_keeps_priority_order(self) -> None:
        best = select_candidate(
            [_result("best_model.zip", 0.8, 10.0), _result("model.zip", 0.8, 10.0)]
        )
        assert best.name == "best_model.zip"

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            select_candidate([])


class TestGatePassed:
    def test_above_threshold_passes(self) -> None:
        assert gate_passed(_result("model.zip", 0.95), 0.90)

    def test_exactly_at_threshold_passes(self) -> None:
        assert gate_passed(_result("model.zip", 0.90), 0.90)

    def test_below_threshold_fails(self) -> None:
        assert not gate_passed(_result("model.zip", 0.89), 0.90)


def test_candidate_from_summary_extracts_metrics(tmp_path: Path) -> None:
    summary = {"success_rate": 0.92, "return_mean": 150.5, "n_episodes": 100, "n_crash": 8}
    cand = candidate_from_summary("best_model.zip", tmp_path / "best_model.zip", summary)
    assert cand.success_rate == 0.92
    assert cand.return_mean == 150.5
    assert cand.n_episodes == 100
    assert cand.summary["n_crash"] == 8
    assert cand.as_report_entry() == {
        "name": "best_model.zip",
        "path": str(tmp_path / "best_model.zip"),
        "success_rate": 0.92,
        "return_mean": 150.5,
        "n_episodes": 100,
    }


class TestWriteGateReport:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "gate_report.json"
        report = {"profile": "progressive", "agents": {"sac": {"status": "completed"}}}
        write_gate_report(path, report)
        assert json.loads(path.read_text()) == report

    def test_atomic_no_tmp_leftover_and_overwrites(self, tmp_path: Path) -> None:
        path = tmp_path / "gate_report.json"
        write_gate_report(path, {"v": 1})
        write_gate_report(path, {"v": 2})
        assert json.loads(path.read_text()) == {"v": 2}
        assert list(tmp_path.iterdir()) == [path]
