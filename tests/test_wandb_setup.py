"""Unit tests for WandB mode resolution.

Covers the precedence rules in :mod:`utils.wandb_setup`: explicit ``WANDB_MODE``
wins, otherwise the mode follows the presence of ``WANDB_API_KEY``.
"""
from __future__ import annotations

import pytest
import wandb
from omegaconf import OmegaConf
from wandb.errors import CommError

from utils.wandb_setup import ensure_project, register_resolvers, resolve_wandb_mode


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Start each test from a clean WandB environment."""
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)


def test_offline_without_key():
    assert resolve_wandb_mode() == "offline"


def test_online_when_key_present(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "deadbeef")
    assert resolve_wandb_mode() == "online"


def test_explicit_mode_overrides_key(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "deadbeef")
    monkeypatch.setenv("WANDB_MODE", "offline")
    assert resolve_wandb_mode() == "offline"


def test_explicit_mode_without_key(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    assert resolve_wandb_mode() == "disabled"


def test_resolver_registered_and_resolves(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "deadbeef")
    register_resolvers()
    cfg = OmegaConf.create({"mode": "${zeta.wandb_mode:}"})
    assert cfg.mode == "online"


class _FakeProject:
    """Minimal stand-in for a wandb ``Project`` (only ``.name`` is used)."""

    def __init__(self, name: str):
        self.name = name


class _FakeApi:
    """Stand-in for ``wandb.Api`` that records calls and never hits the network."""

    default_entity = "test-entity"
    instances: list["_FakeApi"] = []

    #: names returned by ``projects()`` — drives the existence check.
    existing_projects: list[str] = []
    #: when set, ``projects()`` raises this to simulate an auth/network failure.
    projects_raises: Exception | None = None
    #: when set, constructing the API raises this to simulate auth failure.
    init_raises: Exception | None = None

    def __init__(self):
        if type(self).init_raises is not None:
            raise type(self).init_raises
        self.created: list[tuple[str, str]] = []
        self.listed: list[str] = []
        type(self).instances.append(self)

    def projects(self, entity):
        self.listed.append(entity)
        if type(self).projects_raises is not None:
            raise type(self).projects_raises
        return [_FakeProject(n) for n in type(self).existing_projects]

    def create_project(self, name, entity):
        self.created.append((name, entity))


@pytest.fixture
def fake_api(monkeypatch):
    """Patch ``wandb.Api`` with a fresh :class:`_FakeApi` class per test."""

    class FreshApi(_FakeApi):
        instances = []
        existing_projects = []
        projects_raises = None
        init_raises = None

    monkeypatch.setattr(wandb, "Api", FreshApi)
    return FreshApi


def test_ensure_project_skips_when_offline_no_key(fake_api):
    # No WANDB_API_KEY -> mode resolves offline -> no network call, returns None.
    assert ensure_project("zeta-bench") is None
    assert fake_api.instances == []


def test_ensure_project_skips_when_mode_forced_offline(monkeypatch, fake_api):
    monkeypatch.setenv("WANDB_API_KEY", "deadbeef")
    monkeypatch.setenv("WANDB_MODE", "offline")
    assert ensure_project("zeta-bench") is None
    assert fake_api.instances == []


def test_ensure_project_verifies_existing(monkeypatch, fake_api):
    monkeypatch.setenv("WANDB_API_KEY", "deadbeef")
    fake_api.existing_projects = ["sandbox", "zeta-bench"]
    entity = ensure_project("zeta-bench")
    assert entity == "test-entity"
    api = fake_api.instances[0]
    assert api.listed == ["test-entity"]
    assert api.created == []  # verify path: no creation


def test_ensure_project_creates_when_missing(monkeypatch, fake_api):
    monkeypatch.setenv("WANDB_API_KEY", "deadbeef")
    fake_api.existing_projects = ["sandbox"]  # zeta-bench absent
    entity = ensure_project("zeta-bench")
    assert entity == "test-entity"
    api = fake_api.instances[0]
    assert api.created == [("zeta-bench", "test-entity")]


def test_ensure_project_raises_on_auth_failure(monkeypatch, fake_api):
    monkeypatch.setenv("WANDB_API_KEY", "deadbeef")
    fake_api.init_raises = CommError("invalid api key")
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        ensure_project("zeta-bench")
