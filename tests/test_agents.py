"""Integration tests for the SB3 agent wrappers (SAC + PPO).

These require the optional ``train`` extra (``pip install -e '.[train]'``);
they are skipped cleanly via :func:`pytest.importorskip` on machines without
torch / stable-baselines3 (e.g. Intel-mac dev boxes, lean CI). The wrappers
themselves stay importable without the extra (lazy SB3 imports), so the
``import``-level checks below run everywhere.
"""
from __future__ import annotations

import numpy as np
import pytest
from hydra import compose, initialize

from controllers.ppo_agent import PPOAgent
from controllers.sac_agent import SACAgent
from envs.rocket_landing_env import OBS_DIM, RocketLandingEnv

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture
def agent_and_cfg(request):
    """Compose a config scoped to the requested agent class.

    SAC and PPO have disjoint struct fields (e.g. ``n_epochs`` vs
    ``learning_starts``), so each agent must be tested against its own
    config; sharing a single SAC-default config for both causes
    ``ConfigAttributeError`` when PPO accesses its fields.
    """
    agent_cls = request.param
    is_ppo = agent_cls is PPOAgent
    agent_name = "ppo" if is_ppo else "sac"
    overrides = [
        f"agent={agent_name}",
        "compute.n_envs=1",
        "compute.batch_size=8",
        "compute.n_steps=16",
    ]
    if not is_ppo:
        overrides += ["compute.buffer_size=200", "agent.learning_starts=8"]
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="train", overrides=overrides)
    return agent_cls, cfg


def test_agent_modules_import_without_sb3() -> None:
    """Wrappers must be importable even without the train extra installed."""
    assert SACAgent(cfg=None)._model is None  # noqa: SLF001
    assert PPOAgent(cfg=None)._model is None  # noqa: SLF001


def test_predict_before_learn_raises() -> None:
    """Calling predict() before learn()/load() is a clear error, not a crash."""
    with pytest.raises(RuntimeError):
        SACAgent(cfg=None).predict(np.zeros(OBS_DIM))
    with pytest.raises(RuntimeError):
        PPOAgent(cfg=None).predict(np.zeros(OBS_DIM))


@pytest.mark.parametrize("agent_and_cfg", [SACAgent, PPOAgent], indirect=True, ids=["SACAgent", "PPOAgent"])
def test_learn_predict_save_load_roundtrip(agent_and_cfg, tmp_path) -> None:
    """Smoke: train a handful of steps, predict an in-bounds action, and
    round-trip through save/load."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("torch")

    agent_cls, cfg = agent_and_cfg

    # Keep the run tiny and off wandb.
    cfg.total_steps = 64
    cfg.results_dir = str(tmp_path)

    env = RocketLandingEnv(cfg)
    agent = agent_cls(cfg)
    agent.learn(env, total_steps=int(cfg.total_steps))

    obs, _ = env.reset(seed=0)
    action = agent.predict(obs, deterministic=True)
    assert action.shape == (3,)
    assert env.action_space.contains(action.astype(env.action_space.dtype))

    ckpt = tmp_path / "ckpt"
    agent.save(str(ckpt))
    restored = agent_cls.load(str(ckpt))
    action2 = restored.predict(obs, deterministic=True)
    assert action2.shape == (3,)
    np.testing.assert_allclose(action, action2, rtol=1e-5, atol=1e-6)
