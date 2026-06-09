"""Experiment entrypoints.

Orchestration scripts only — no business logic, no shared utilities. Any
helper code should live in ``utils/`` or the appropriate domain package.

Import rule: this package is a leaf in the dependency graph. Nothing else
in the project may import from ``experiments/``.
"""
