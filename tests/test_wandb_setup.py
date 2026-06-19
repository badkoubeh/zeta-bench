"""Unit tests for WandB mode resolution.

Covers the precedence rules in :mod:`utils.wandb_setup`: explicit ``WANDB_MODE``
wins, otherwise the mode follows the presence of ``WANDB_API_KEY``.
"""
from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from utils.wandb_setup import register_resolvers, resolve_wandb_mode


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
