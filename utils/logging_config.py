"""Python logging setup.

Single entry point: :func:`get_logger`. The root logger is configured on the
first call (idempotent) with a project-standard format string. Never use
``print`` in project code.
"""
from __future__ import annotations

import logging
import sys

_CONFIGURED: bool = False
_FORMAT: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for ``name``. Idempotent."""
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(level=logging.INFO, format=_FORMAT, stream=sys.stdout)
        _CONFIGURED = True
    return logging.getLogger(name)
