"""Smoke tests — every top-level package imports cleanly.

Phase 1 minimum bar. These tests catch broken imports, circular dependencies,
and Gymnasium registration errors. They do NOT verify physics, reward, or
training correctness; those land in later phases.
"""
from __future__ import annotations


def test_packages_importable() -> None:
    """All top-level packages import without raising."""
    import adversary  # noqa: F401
    import controllers  # noqa: F401
    import dynamics  # noqa: F401
    import envs  # noqa: F401
    import experiments  # noqa: F401
    import utils  # noqa: F401


def test_gym_env_registered() -> None:
    """The Gymnasium env is registered under the expected ID."""
    import envs  # noqa: F401  # triggers register()
    from gymnasium.envs.registration import registry

    assert "RocketLanding-v0" in registry, "RocketLanding-v0 not registered"
