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
    """Verify wandb auth and resolve the target entity for an online run.

    A preflight guard for online runs: it authenticates the configured
    ``WANDB_API_KEY`` and resolves the entity (team/user) the run will log to.
    wandb already auto-creates a project on the first ``wandb.init``, so this
    does not make logging *possible* — its value is failing fast with a clear
    message when a key is missing/invalid, rather than silently logging offline
    or into an unexpected entity.

    Auth is verified by constructing :class:`wandb.Api` with the key (which
    validates it and raises on a bad key) and resolving the default entity. We
    deliberately avoid ``api.projects()``/``api.create_project()``: in wandb
    0.27.x the Public API pagination path through wandb-core spuriously fails
    with "relogin required" even for valid credentials, while ``wandb.init``
    logs fine and auto-creates the project. Project creation is therefore left
    to ``wandb.init``.

    Parameters
    ----------
    project:
        Project name the run will log to (e.g. ``cfg.wandb.project``). Used for
        logging context only; creation is handled by ``wandb.init``.
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
        If a key is present but authentication fails.
    """
    if resolve_wandb_mode() != "online" or not os.environ.get("WANDB_API_KEY"):
        _logger.debug("wandb not online or no API key; skipping project preflight")
        return None

    import wandb

    def _verify_auth(api_key: str | None = None) -> str:
        # Constructing Api with the key validates it (raises on a bad key);
        # resolving the entity confirms which account we are pointed at. This
        # uses the lightweight viewer/default-entity path, not the broken
        # projects() pagination.
        api = wandb.Api(api_key=api_key) if api_key else wandb.Api()
        return entity or api.default_entity

    try:
        resolved_entity = _verify_auth()
    except Exception as exc:  # auth / network failures surface here
        # A cached wandb session can expire while WANDB_API_KEY is still valid;
        # wandb-core reports this as "relogin required". Rather than abort, force
        # a fresh login from the key and retry once before giving up. Other
        # failures (invalid key, network) fall through to the hard error.
        if "relogin" in str(exc).lower() and os.environ.get("WANDB_API_KEY"):
            _logger.warning(
                "wandb session expired (relogin required); re-authenticating "
                "from WANDB_API_KEY and retrying"
            )
            try:
                wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
                # The persistent wandb-core service keeps its own (now stale)
                # session that wandb.login does not refresh; drop it so the
                # retried Api spins up a service with fresh credentials.
                try:
                    wandb.teardown()
                except Exception:  # best-effort; never mask the real error
                    pass
                resolved_entity = _verify_auth(api_key=os.environ["WANDB_API_KEY"])
            except Exception as retry_exc:
                raise RuntimeError(
                    "wandb authentication failed after a forced relogin; "
                    "check WANDB_API_KEY (get a key at "
                    "https://wandb.ai/authorize)"
                ) from retry_exc
        else:
            raise RuntimeError(
                "wandb authentication failed; check WANDB_API_KEY "
                "(get a key at https://wandb.ai/authorize)"
            ) from exc

    _logger.info(
        "wandb auth verified; run will log to %s/%s "
        "(project auto-created on init if missing)",
        resolved_entity,
        project,
    )
    return resolved_entity


def register_resolvers() -> None:
    """Register the ``${zeta.wandb_mode:}`` OmegaConf resolver (idempotent).

    Must be called before Hydra resolves any config that references the resolver.
    Entry points call this at import time.
    """
    OmegaConf.register_new_resolver(
        _RESOLVER_NAME, resolve_wandb_mode, replace=True
    )
