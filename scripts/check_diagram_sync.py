#!/usr/bin/env python3
"""Pre-commit guard: block architecture changes that don't update the diagram.

The closed-loop control diagram in ``README.md`` is the authoritative visual
specification of the zeta-bench system. If any of the trigger paths below are
staged for commit without ``README.md`` also being staged, this hook fails
the commit.

Bypass (only when the change does NOT affect signal flow):

    SKIP_DIAGRAM_CHECK=1 git commit ...

Stdlib only — runs under whatever Python the git hook environment provides.
"""
from __future__ import annotations

import os
import subprocess
import sys

TRIGGER_DIRS: tuple[str, ...] = (
    "dynamics/",
    "controllers/",
    "adversary/",
)
TRIGGER_FILES: frozenset[str] = frozenset(
    {
        "envs/rocket_landing_env.py",
        "envs/__init__.py",
        "configs/env.yaml",
        "configs/reward.yaml",
        "configs/adversary.yaml",
    }
)
README: str = "README.md"


def staged_files() -> list[str]:
    """Return paths in the current commit's index (added/copied/modified/renamed)."""
    proc = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in proc.stdout.splitlines() if line]


def is_trigger(path: str) -> bool:
    """True if `path` is one of the architecture-relevant trigger paths."""
    if path in TRIGGER_FILES:
        return True
    return path.endswith(".py") and any(path.startswith(d) for d in TRIGGER_DIRS)


def main() -> int:
    if os.environ.get("SKIP_DIAGRAM_CHECK") == "1":
        return 0

    files = staged_files()
    triggers = [f for f in files if is_trigger(f)]
    if not triggers:
        return 0
    if README in files:
        return 0

    bar = "=" * 62
    print(f"\n{bar}", file=sys.stderr)
    print(" Closed-loop diagram out of sync?", file=sys.stderr)
    print(bar, file=sys.stderr)
    print(
        " You staged architecture-relevant files but did NOT update\n"
        " README.md (which carries the closed-loop control diagram).\n\n"
        " Staged trigger files:",
        file=sys.stderr,
    )
    for f in triggers:
        print(f"   - {f}", file=sys.stderr)
    print(
        "\n If signal flow, action space, obs space, or dynamics changed,\n"
        " update the diagram in README.md before committing.\n\n"
        " If this change does NOT affect signal flow, bypass with:\n"
        "   SKIP_DIAGRAM_CHECK=1 git commit ...",
        file=sys.stderr,
    )
    print(f"{bar}\n", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
