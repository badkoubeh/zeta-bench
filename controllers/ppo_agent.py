"""PPO agent wrapper around Stable-Baselines3.

Thin adapter exposing the project's uniform agent interface (``predict``,
``save``, ``load``, ``learn``), structurally identical to
:class:`controllers.sac_agent.SACAgent` — same vectorised-env construction,
same compute-profile overrides, same wandb logging callback. Hyperparameters
come from ``configs/agent/ppo.yaml``.

Stable-Baselines3 and torch are imported **lazily** inside :meth:`learn` /
:meth:`load` so this module stays importable without the optional ``train``
extra (``pip install -e '.[train]'``).
"""
from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from utils.logging_config import get_logger

logger = get_logger(__name__)


class PPOAgent:
    """SB3 PPO wrapper with project-standard logging."""

    def __init__(self, cfg: DictConfig | None = None) -> None:
        """Construct from a composed config; defer SB3 model creation to ``learn``.

        ``cfg`` may be ``None`` when rebuilt by :meth:`load`.
        """
        self._cfg = cfg
        self._model: object | None = None

    def _resolve_device(self, requested: str) -> str:
        """Return the usable device, falling back to CPU if the accelerator is absent."""
        import torch

        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning("compute.device=cuda requested but CUDA unavailable; using cpu")
            return "cpu"
        if requested == "mps" and not torch.backends.mps.is_available():
            logger.warning("compute.device=mps requested but MPS unavailable; using cpu")
            return "cpu"
        return requested

    def learn(self, env: object, total_steps: int) -> None:
        """Train against ``env`` for ``total_steps`` environment steps.

        Builds a vectorised env (``compute.n_envs`` parallel workers) from the
        config, attaches the wandb logging + checkpoint callbacks, and runs
        SB3 PPO. With ``compute.n_envs > 1`` the workers are fresh env copies
        built from ``cfg``; the passed ``env`` is used only for the single-env
        case (e.g. tests).
        """
        if self._cfg is None:
            raise RuntimeError("PPOAgent.learn requires a config (was the agent loaded?)")

        from omegaconf import OmegaConf
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
        from stable_baselines3.common.env_util import make_vec_env

        from utils.sb3_callbacks import WandbLoggingCallback
        from envs.rocket_landing_env import RocketLandingEnv

        cfg = self._cfg
        a = cfg.agent
        compute = cfg.get("compute", None)
        device = self._resolve_device(str(compute.device) if compute else "cpu")
        n_envs = int(compute.n_envs) if compute else 1
        batch_size = int(compute.batch_size) if compute else int(a.batch_size)
        n_steps = int(compute.n_steps) if compute else int(a.n_steps)
        seed = int(cfg.seed)

        if n_envs > 1:
            vec_env = make_vec_env(lambda: RocketLandingEnv(cfg), n_envs=n_envs, seed=seed)
        else:
            vec_env = make_vec_env(
                lambda: env if env is not None else RocketLandingEnv(cfg),
                n_envs=1,
                seed=seed,
            )

        resume_from = cfg.get("resume_from", None)
        if resume_from:
            logger.info("resuming PPO from checkpoint %s", resume_from)
            self._model = PPO.load(str(resume_from), env=vec_env, device=device)
        else:
            self._model = PPO(
                policy=str(a.policy),
                env=vec_env,
                learning_rate=float(a.learning_rate),
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=int(a.n_epochs),
                gamma=float(a.gamma),
                gae_lambda=float(a.gae_lambda),
                clip_range=float(a.clip_range),
                ent_coef=float(a.ent_coef),
                vf_coef=float(a.vf_coef),
                max_grad_norm=float(a.max_grad_norm),
                policy_kwargs={"net_arch": list(a.policy_kwargs.net_arch)},
                device=device,
                seed=seed,
                verbose=1,
            )

        callbacks = [WandbLoggingCallback(cfg)]
        ckpt = cfg.get("checkpoint", None)
        if ckpt is not None and "results_dir" in cfg:
            save_freq = max(int(ckpt.every_n_steps) // max(n_envs, 1), 1)
            callbacks.append(
                CheckpointCallback(
                    save_freq=save_freq,
                    save_path=str(cfg.results_dir),
                    name_prefix=str(cfg.run_name),
                )
            )

        if "results_dir" in cfg:
            # Eval env: separate env with curriculum pinned to a fixed difficulty so
            # best_model.zip is selected consistently rather than on the (annealing)
            # training distribution. Driven by eval_callback.task_difficulty
            # (default 1.0 when absent for back-compat); set to 0.0 to select on the
            # pure-vertical regime that matches the PID baseline.
            evalcb = cfg.get("eval_callback", None)
            eval_task_difficulty = (
                float(evalcb.task_difficulty)
                if evalcb is not None and "task_difficulty" in evalcb
                else 1.0
            )
            eval_cfg = OmegaConf.merge(
                cfg,
                OmegaConf.create(
                    {
                        "env": {
                            "curriculum": {
                                "schedule": "fixed",
                                "task_difficulty": eval_task_difficulty,
                            }
                        }
                    }
                ),
            )
            eval_env = make_vec_env(lambda: RocketLandingEnv(eval_cfg), n_envs=1, seed=seed + 999)
            eval_every = int(evalcb.every_n_steps) if evalcb is not None else 50_000
            n_eval_eps = int(evalcb.n_eval_episodes) if evalcb is not None else 20
            # eval_freq is counted in vec-env steps, so scale by n_envs to keep the
            # cadence fixed in env steps regardless of the compute profile's worker count.
            eval_freq = max(eval_every // max(n_envs, 1), 1)
            callbacks.append(
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(cfg.results_dir),
                    log_path=str(cfg.results_dir),
                    eval_freq=eval_freq,
                    n_eval_episodes=n_eval_eps,
                    deterministic=True,
                    verbose=1,
                )
            )

        steps_done = getattr(self._model, "num_timesteps", 0) if resume_from else 0
        # `total_steps` is the number of NEW environment steps to run, not a
        # cumulative cap. On a fresh run that is the full budget; when resuming,
        # SB3 re-adds the loaded num_timesteps internally (because
        # reset_num_timesteps=False), so passing this value straight through
        # trains exactly `total_steps` additional steps on top of the checkpoint.
        # (The old cumulative semantics silently no-op'd once a checkpoint
        # already exceeded the requested total.)
        new_steps = max(int(total_steps), 0)
        reset_num_timesteps = not bool(resume_from)

        logger.info(
            "PPO.learn: new_steps=%d steps_done=%d target_total=%d n_envs=%d device=%s n_steps=%d batch=%d",
            new_steps,
            steps_done,
            steps_done + new_steps,
            n_envs,
            device,
            n_steps,
            batch_size,
        )
        if new_steps == 0:
            logger.info("total_steps resolved to 0 new steps; nothing to train")
            return
        self._model.learn(
            total_timesteps=new_steps,
            callback=CallbackList(callbacks),
            reset_num_timesteps=reset_num_timesteps,
        )

        if "results_dir" in cfg:
            final_path = f"{cfg.results_dir}/model"
            self._model.save(final_path)
            logger.info("saved final PPO model to %s.zip", final_path)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Compute a 3-dim action from a 17-dim observation."""
        if self._model is None:
            raise RuntimeError("PPOAgent.predict called before learn()/load()")
        action, _ = self._model.predict(np.asarray(obs), deterministic=deterministic)
        return np.asarray(action, dtype=np.float64)

    def save(self, path: str) -> None:
        """Persist the SB3 model to disk (SB3 appends ``.zip``)."""
        if self._model is None:
            raise RuntimeError("PPOAgent.save called before learn()/load()")
        self._model.save(path)

    @classmethod
    def load(cls, path: str) -> "PPOAgent":
        """Restore agent from a saved SB3 checkpoint."""
        from stable_baselines3 import PPO

        agent = cls(cfg=None)
        agent._model = PPO.load(path)
        return agent
