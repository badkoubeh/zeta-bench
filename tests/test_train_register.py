"""Unit tests for ``experiments.train._register_model``.

Focus: the registration step must be **best-effort** — when the model file
exists and the run is online it logs a session-tagged artifact and links it into
the Model Registry, but any W&B failure is swallowed (logged, never raised) so an
otherwise-successful training run does not exit non-zero. Offline runs skip the
network entirely.
"""
from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from experiments import train


class _FakeArtifact:
    """Captures the args ``_register_model`` passes to ``wandb.Artifact``."""

    def __init__(self, name, type, metadata):  # noqa: A002 - mirror wandb API
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files: list[tuple[str, str]] = []

    def add_file(self, path, name):
        self.files.append((path, name))


class _FakeRun:
    """Minimal stand-in for ``wandb.run`` recording log/link calls."""

    id = "abc123"
    name = "sac_moderate_nominal_42"
    path = ["test-entity", "zeta-bench", "abc123"]
    url = "https://wandb.ai/test-entity/zeta-bench/runs/abc123"
    entity = "test-entity"
    project = "zeta-bench"

    def __init__(self, *, link_raises: bool = False):
        self._link_raises = link_raises
        self.summary: dict[str, object] = {}
        self.logged: list[_FakeArtifact] = []
        self.linked: list[tuple[object, str, list[str]]] = []

    def log_artifact(self, artifact):
        self.logged.append(artifact)
        return artifact  # the real API returns a handle exposing .wait()

    def link_artifact(self, artifact, target_path, aliases):
        if self._link_raises:
            raise RuntimeError("registry unreachable")
        self.linked.append((artifact, target_path, aliases))


class _FakeArtifactHandle:
    """``log_artifact`` returns this; ``.wait()`` must be a no-op here."""


def _cfg(mode: str = "online"):
    return OmegaConf.create(
        {
            "seed": 42,
            "total_steps": 2_000_000,
            "train_mode": "nominal",
            "env": {"dynamics": {"fidelity": "moderate"}},
            "wandb": {"mode": mode},
        }
    )


@pytest.fixture
def results_dir(tmp_path):
    """A results dir containing a best_model.zip, as a real run would leave."""
    (tmp_path / "best_model.zip").write_bytes(b"fake-model")
    return tmp_path


def _install_fake_wandb(monkeypatch, run):
    fake_wandb = type("W", (), {})()
    fake_wandb.run = run
    fake_wandb.Artifact = _FakeArtifact
    monkeypatch.setattr(train, "wandb", fake_wandb)


def test_register_tags_session_and_links(monkeypatch, results_dir):
    run = _FakeRun()
    # log_artifact returns a handle with a .wait(); patch in a stub.
    handle = _FakeArtifactHandle()
    handle.wait = lambda: None  # type: ignore[attr-defined]
    run.log_artifact = lambda artifact: (run.logged.append(artifact) or handle)
    _install_fake_wandb(monkeypatch, run)

    train._register_model(_cfg("online"), "sac", results_dir)

    assert run.summary["model_registered"] is True
    artifact = run.logged[0]
    # Session-identifying metadata is present and points back to the run.
    assert artifact.metadata["run_id"] == "abc123"
    assert artifact.metadata["run_url"].endswith("/runs/abc123")
    assert artifact.metadata["run_path"] == "test-entity/zeta-bench/abc123"
    assert artifact.metadata["agent"] == "sac"
    assert artifact.name == "zetabench-sac"
    # Linked into the registry with a `best` alias.
    _, target, aliases = run.linked[0]
    assert target == "wandb-registry-model/zetabench-sac"
    assert "best" in aliases


def test_register_failure_is_not_fatal(monkeypatch, results_dir):
    run = _FakeRun(link_raises=True)
    handle = _FakeArtifactHandle()
    handle.wait = lambda: None  # type: ignore[attr-defined]
    run.log_artifact = lambda artifact: (run.logged.append(artifact) or handle)
    _install_fake_wandb(monkeypatch, run)

    # Must not raise even though link_artifact blows up.
    train._register_model(_cfg("online"), "sac", results_dir)

    assert run.summary["model_registered"] is False


def test_register_offline_skips_network(monkeypatch, results_dir):
    run = _FakeRun()
    _install_fake_wandb(monkeypatch, run)

    train._register_model(_cfg("offline"), "sac", results_dir)

    assert run.logged == []
    assert run.linked == []
    assert "model_registered" not in run.summary


def test_register_no_model_file_noops(monkeypatch, tmp_path):
    run = _FakeRun()
    _install_fake_wandb(monkeypatch, run)

    train._register_model(_cfg("online"), "sac", tmp_path)

    assert run.logged == []
