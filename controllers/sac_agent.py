"""SAC agent wrapper around Stable-Baselines3.

Thin adapter exposing the project's uniform agent interface (``predict``,
``save``, ``load``, ``learn``), matching :class:`controllers.pid_baseline.PIDController`
so the evaluation harness can drive any controller identically. Hyperparameters
come from ``configs/agent/sac.yaml``; the compute profile (``configs/compute/*.yaml``)
overrides throughput-shaping fields and selects the device.

The training callback (:class:`utils.sb3_callbacks.WandbLoggingCallback`)
logs every reward component, curriculum progress, and episode-level metrics to
wandb separately from the total reward (per ``CONTRIBUTING.md`` §Experiment Tracking).

Stable-Baselines3 and torch are imported **lazily** inside :meth:`learn` /
:meth:`load` so this module stays importable without the optional ``train``
extra (``pip install -e '.[train]'``).
"""
from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from utils.logging_config import get_logger

logger = get_logger(__name__)


class SACAgent:
    """SB3 SAC wrapper with project-standard logging."""

    def __init__(self, cfg: DictConfig | None = None) -> None:
        """Construct from a composed config; defer SB3 model creation to ``learn``.

        ``cfg`` may be ``None`` when the instance is being rebuilt by
        :meth:`load` (the SB3 checkpoint carries its own hyperparameters).
        """
        self._cfg = cfg
        self._model: object | None = None

    def _resolve_device(self, requested: str) -> str:
        """Return the usable device, falling back to CPU if the accelerator is absent.

        Lets the same config run locally on CPU, on Apple Silicon (``compute=mps``),
        and on a CUDA cloud instance (``compute=small_gpu`` etc.) without code changes,
        while failing loudly in logs rather than silently if a GPU was requested but is
        unavailable.
        """
        import torch

        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning("compute.device=cuda requested but CUDA unavailable; using cpu")
            return "cpu"
        if requested == "mps" and not torch.backends.mps.is_available():
            logger.warning("compute.device=mps requested but MPS unavailable; using cpu")
            return "cpu"
        return requested

    @staticmethod
    def _find_replay_buffer(checkpoint: "object") -> "object | None":
        """Locate a saved replay buffer next to ``checkpoint`` (a ``Path``).

        Prefers a sibling ``replay_buffer.pkl`` (written at end of training),
        otherwise the most recent ``*replay_buffer*.pkl`` in the same folder.
        Returns the ``Path`` to use, or ``None`` when no buffer is available
        (e.g. resuming from an ``EvalCallback`` ``best_model.zip`` in a run that
        predates buffer persistence).
        """
        from pathlib import Path

        ckpt = Path(checkpoint)
        sibling = ckpt.parent / "replay_buffer.pkl"
        if sibling.exists():
            return sibling
        candidates = sorted(
            ckpt.parent.glob("*replay_buffer*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def learn(self, env: object, total_steps: int) -> None:
        """Train against ``env`` for ``total_steps`` environment steps.

        Builds a vectorised env (``compute.n_envs`` parallel workers) from the
        config, attaches the wandb logging + checkpoint callbacks, and runs
        SB3 SAC. With ``compute.n_envs > 1`` the workers are fresh env copies
        built from ``cfg``; the passed ``env`` is used only for the single-env
        case (e.g. tests).
        """
        if self._cfg is None:
            raise RuntimeError("SACAgent.learn requires a config (was the agent loaded?)")

        from omegaconf import OmegaConf
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
        from stable_baselines3.common.env_util import make_vec_env

        from utils.sb3_callbacks import WandbLoggingCallback
        from envs.rocket_landing_env import RocketLandingEnv
        from envs.domain_randomization import wrap_if_enabled

        cfg = self._cfg
        a = cfg.agent
        compute = cfg.get("compute", None)
        device = self._resolve_device(str(compute.device) if compute else "cpu")
        n_envs = int(compute.n_envs) if compute else 1
        batch_size = int(compute.batch_size) if compute else int(a.batch_size)
        buffer_size = int(compute.buffer_size) if compute else int(a.buffer_size)
        gradient_steps = int(compute.gradient_steps) if (compute and "gradient_steps" in compute) else int(a.gradient_steps)
        seed = int(cfg.seed)

        # Training env only: optionally apply domain randomisation so the policy
        # learns across the disturbance distribution. The eval env below stays
        # nominal so model selection is on the clean task.
        if n_envs > 1:
            vec_env = make_vec_env(
                lambda: wrap_if_enabled(RocketLandingEnv(cfg), cfg), n_envs=n_envs, seed=seed
            )
        else:
            vec_env = make_vec_env(
                lambda: wrap_if_enabled(
                    env if env is not None else RocketLandingEnv(cfg), cfg
                ),
                n_envs=1,
                seed=seed,
            )

        resume_from = cfg.get("resume_from", None)
        if resume_from:
            from pathlib import Path

            logger.info("resuming SAC from checkpoint %s", resume_from)
            self._model = SAC.load(str(resume_from), env=vec_env, device=device)

            # SAC.load restores only the policy/optimizer state — the replay
            # buffer is saved separately. Without it, num_timesteps is already
            # past learning_starts, so the first learn() step runs gradient
            # updates against an empty buffer and destroys the loaded policy.
            # Restore the buffer if one sits next to the checkpoint; otherwise
            # force a fresh warmup so SAC re-collects transitions before any
            # gradient updates.
            buffer_path = self._find_replay_buffer(Path(str(resume_from)))
            if buffer_path is not None:
                logger.info("restoring replay buffer from %s", buffer_path)
                self._model.load_replay_buffer(str(buffer_path))
            else:
                warmup = int(self._model.num_timesteps) + int(a.learning_starts)
                logger.warning(
                    "no replay buffer found next to %s; forcing a %d-step warmup "
                    "before gradient updates to avoid empty-buffer policy collapse",
                    resume_from,
                    int(a.learning_starts),
                )
                self._model.learning_starts = warmup
        else:
            self._model = SAC(
                policy=str(a.policy),
                env=vec_env,
                learning_rate=float(a.learning_rate),
                buffer_size=buffer_size,
                batch_size=batch_size,
                gamma=float(a.gamma),
                tau=float(a.tau),
                train_freq=int(a.train_freq),
                gradient_steps=gradient_steps,
                learning_starts=int(a.learning_starts),
                ent_coef=a.ent_coef,
                target_entropy=a.target_entropy,
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
            "SAC.learn: new_steps=%d steps_done=%d target_total=%d n_envs=%d device=%s batch=%d buffer=%d grad_steps=%d",
            new_steps,
            steps_done,
            steps_done + new_steps,
            n_envs,
            device,
            batch_size,
            buffer_size,
            gradient_steps,
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
            # Persist the replay buffer so a later run can resume_from this
            # checkpoint with a warm buffer instead of collapsing / re-warming.
            self._model.save_replay_buffer(f"{cfg.results_dir}/replay_buffer")
            logger.info("saved final SAC model to %s.zip (+ replay_buffer.pkl)", final_path)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Compute a 3-dim action from a 17-dim observation."""
        if self._model is None:
            raise RuntimeError("SACAgent.predict called before learn()/load()")
        action, _ = self._model.predict(np.asarray(obs), deterministic=deterministic)
        return np.asarray(action, dtype=np.float64)

    def save(self, path: str) -> None:
        """Persist the SB3 model to disk (SB3 appends ``.zip``)."""
        if self._model is None:
            raise RuntimeError("SACAgent.save called before learn()/load()")
        self._model.save(path)

    @classmethod
    def load(cls, path: str) -> "SACAgent":
        """Restore agent from a saved SB3 checkpoint."""
        from stable_baselines3 import SAC

        agent = cls(cfg=None)
        agent._model = SAC.load(path)
        return agent
