"""WandB mode resolution.

The Weights & Biases logging mode is derived from the environment rather than
hardcoded, so a single config and a single image behave correctly whether or not
an API key is present:

1. If ``WANDB_MODE`` is explicitly set (non-empty), honour it verbatim — this is
   the manual override (e.g. force ``offline`` on a CI box that *does* have a key).
2. Otherwise, default to ``online`` when a ``WANDB_API_KEY`` is available, and
   fall back to ``offline`` when it is not.

Step 2 makes "drop your key in ``.env`` and runs are tracked" true without a
second opt-in step, while keeping unauthenticated runs from blocking on a login
prompt.
"""
from __future__ import annotations

import os

from omegaconf import OmegaConf

_RESOLVER_NAME = "zeta.wandb_mode"


def resolve_wandb_mode() -> str:
    """Return the WandB mode implied by the current environment.

    Returns
    -------
    str
        ``"online"``, ``"offline"``, or whatever non-empty value ``WANDB_MODE``
        was explicitly set to. See the module docstring for the precedence rules.
    """
    explicit = os.environ.get("WANDB_MODE")
    if explicit:
        return explicit
    return "online" if os.environ.get("WANDB_API_KEY") else "offline"


def register_resolvers() -> None:
    """Register the ``${zeta.wandb_mode:}`` OmegaConf resolver (idempotent).

    Must be called before Hydra resolves any config that references the resolver.
    Entry points call this at import time.
    """
    OmegaConf.register_new_resolver(
        _RESOLVER_NAME, resolve_wandb_mode, replace=True
    )
