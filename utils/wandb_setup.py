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

from utils.logging_config import get_logger

_RESOLVER_NAME = "zeta.wandb_mode"

_logger = get_logger(__name__)


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


def ensure_project(project: str, *, entity: str | None = None) -> str | None:
    """Verify wandb auth and ensure ``project`` exists, creating it if missing.

    A preflight guard for online runs: it authenticates the configured
    ``WANDB_API_KEY`` and confirms the target project is reachable in the
    resolved entity, creating it when it cannot be fetched. wandb already
    auto-creates a project on the first ``wandb.init``, so this does not make
    logging *possible* — its value is failing fast with a clear message when a
    key is missing/invalid or pointed at the wrong account, rather than silently
    logging offline or into an unexpected entity.

    Parameters
    ----------
    project:
        Project name to verify or create (e.g. ``cfg.wandb.project``).
    entity:
        Target entity (team/user). Defaults to the API key's default entity.

    Returns
    -------
    str | None
        The resolved entity on success, or ``None`` when the run is not online
        (offline/disabled mode, or no API key) — in which case no network call
        is made, preserving the "offline just works" contract used by CI/tests.

    Raises
    ------
    RuntimeError
        If a key is present but authentication or the project operation fails.
    """
    if resolve_wandb_mode() != "online" or not os.environ.get("WANDB_API_KEY"):
        _logger.debug("wandb not online or no API key; skipping project preflight")
        return None

    import wandb
    from wandb.errors import Error

    try:
        api = wandb.Api()
        resolved_entity = entity or api.default_entity
        # NB: api.project() is lazy — it constructs a Project object without
        # querying, so it never signals a missing project. Listing projects
        # actually hits the API, so membership here reflects reality.
        existing = {p.name for p in api.projects(resolved_entity)}
    except Exception as exc:  # auth / network failures surface here
        raise RuntimeError(
            "wandb authentication failed; check WANDB_API_KEY "
            "(get a key at https://wandb.ai/authorize)"
        ) from exc

    if project in existing:
        _logger.info("wandb project verified: %s/%s", resolved_entity, project)
    else:
        _logger.info(
            "wandb project %s/%s not found; creating it", resolved_entity, project
        )
        try:
            api.create_project(name=project, entity=resolved_entity)
        except Error as create_exc:
            raise RuntimeError(
                f"failed to create wandb project {resolved_entity}/{project}; "
                "check WANDB_API_KEY permissions"
            ) from create_exc
        _logger.info("wandb project created: %s/%s", resolved_entity, project)

    return resolved_entity


def register_resolvers() -> None:
    """Register the ``${zeta.wandb_mode:}`` OmegaConf resolver (idempotent).

    Must be called before Hydra resolves any config that references the resolver.
    Entry points call this at import time.
    """
    OmegaConf.register_new_resolver(
        _RESOLVER_NAME, resolve_wandb_mode, replace=True
    )
