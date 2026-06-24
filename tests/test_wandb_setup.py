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


class _FakeApi:
    """Stand-in for ``wandb.Api`` that records calls and never hits the network."""

    instances: list["_FakeApi"] = []

    #: value returned by the ``default_entity`` property.
    default_entity_value: str = "test-entity"
    #: when set, ``default_entity`` raises this to simulate an auth/session error.
    default_entity_raises: Exception | None = None
    #: when set, constructing the API raises this to simulate auth failure.
    init_raises: Exception | None = None

    def __init__(self, api_key=None):
        if type(self).init_raises is not None:
            raise type(self).init_raises
        self.api_key = api_key
        type(self).instances.append(self)

    @property
    def default_entity(self):
        if type(self).default_entity_raises is not None:
            raise type(self).default_entity_raises
        return type(self).default_entity_value


@pytest.fixture
def fake_api(monkeypatch):
    """Patch ``wandb.Api`` with a fresh :class:`_FakeApi` class per test."""

    class FreshApi(_FakeApi):
        instances = []
        default_entity_value = "test-entity"
        default_entity_raises = None
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


def test_ensure_project_returns_default_entity(monkeypatch, fake_api):
    monkeypatch.setenv("WANDB_API_KEY", "deadbeef")
    entity = ensure_project("zeta-bench")
    assert entity == "test-entity"
    # Auth is verified by constructing the Api once; no project listing.
    assert len(fake_api.instances) == 1


def test_ensure_project_uses_explicit_entity(monkeypatch, fake_api):
    monkeypatch.setenv("WANDB_API_KEY", "deadbeef")
    # An explicit entity is returned as-is, without resolving the default.
    fake_api.default_entity_raises = RuntimeError("default_entity must not be read")
    entity = ensure_project("zeta-bench", entity="my-team")
    assert entity == "my-team"


def test_ensure_project_raises_on_auth_failure(monkeypatch, fake_api):
    monkeypatch.setenv("WANDB_API_KEY", "deadbeef")
    fake_api.init_raises = CommError("invalid api key")
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        ensure_project("zeta-bench")


def test_ensure_project_relogins_on_expired_session(monkeypatch, fake_api):
    # A stale session surfaces as "relogin required"; ensure_project should
    # force a fresh login from the key and retry once, then succeed.
    monkeypatch.setenv("WANDB_API_KEY", "deadbeef")
    fake_api.default_entity_raises = CommError("relogin required")

    calls = []

    def fake_login(key, relogin):
        calls.append((key, relogin))
        fake_api.default_entity_raises = None  # session restored after relogin

    monkeypatch.setattr(wandb, "login", fake_login)
    monkeypatch.setattr(wandb, "teardown", lambda: None)

    assert ensure_project("zeta-bench") == "test-entity"
    assert calls == [("deadbeef", True)]
    # The retry must authenticate explicitly with the env key so the new Api
    # carries fresh credentials rather than the expired service session.
    assert fake_api.instances[-1].api_key == "deadbeef"


def test_ensure_project_relogin_retry_still_fails(monkeypatch, fake_api):
    # If the forced relogin does not fix things, surface a clear hard error.
    monkeypatch.setenv("WANDB_API_KEY", "deadbeef")
    fake_api.default_entity_raises = CommError("relogin required")
    monkeypatch.setattr(wandb, "login", lambda key, relogin: None)
    monkeypatch.setattr(wandb, "teardown", lambda: None)

    with pytest.raises(RuntimeError, match="after a forced relogin"):
        ensure_project("zeta-bench")
