"""Shared Stable-Baselines3 training callback for wandb logging.

Both :class:`controllers.sac_agent.SACAgent` and
:class:`controllers.ppo_agent.PPOAgent` attach this callback during
``learn`` so the two algorithms log an identical metric schema (per
``CONTRIBUTING.md`` §Experiment Tracking):

- **Every reward component separately** (not just the scalar total), read
  from ``info["reward_components"]`` emitted by
  :meth:`envs.rocket_landing_env.RocketLandingEnv.step`.
- **Curriculum task difficulty** (current annealed initial-condition difficulty).
- **Per-episode outcome metrics**: landing-success / crash / out-of-bounds /
  timeout, touchdown velocity, and fuel used.

This module imports Stable-Baselines3 at module load, so it is only imported
*lazily* (inside ``learn``) by the agent wrappers — keeping those wrappers
importable on machines without the optional ``train`` extra installed.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import wandb
from omegaconf import DictConfig
from stable_baselines3.common.callbacks import BaseCallback

from utils.normalisation import FixedObsScaler


class WandbLoggingCallback(BaseCallback):
    """Log reward decomposition, curriculum task difficulty, and episode outcomes.

    Component rewards are accumulated and logged as running means every
    ``log_freq`` environment steps to keep wandb traffic bounded on long
    (multi-million-step) runs. Episode-outcome metrics are logged the moment
    an episode terminates, using the vectorised env's ``terminal_observation``
    to recover touchdown velocity and fuel used in physical units.

    Parameters
    ----------
    cfg : DictConfig
        Composed Hydra config (needs ``env`` for the obs-scaler bounds and
        ``env.dynamics.initial_fuel_kg``).
    log_freq : int
        Environment-step interval between component-mean log flushes.
    """

    def __init__(self, cfg: DictConfig, log_freq: int = 1000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._cfg = cfg
        self._scaler = FixedObsScaler(cfg)
        self._initial_fuel_kg = float(cfg.env.dynamics.initial_fuel_kg)
        self._log_freq = int(log_freq)

        self._comp_sums: dict[str, float] = defaultdict(float)
        self._comp_count: int = 0
        self._steps_since_flush: int = 0
        self._latest_progress: float = 0.0
        # Rolling window of recent episode outcomes for a smooth success rate
        # that is readable regardless of how many episodes end per logging step.
        self._outcome_window: deque[str] = deque(maxlen=100)

    def _wandb_active(self) -> bool:
        """True when a wandb run is live (skip logging in tests / dry runs)."""
        return wandb.run is not None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", []) or []
        dones = self.locals.get("dones", [])

        finished: list[dict] = []
        for i, info in enumerate(infos):
            components = info.get("reward_components")
            if components:
                for key, value in components.items():
                    self._comp_sums[key] += float(value)
                self._comp_count += 1
            if "task_difficulty" in info:
                self._latest_task_difficulty = float(info["task_difficulty"])

            done = bool(dones[i]) if i < len(dones) else False
            if done:
                finished.append(info)

        if finished:
            self._log_episode_outcomes(finished)

        self._steps_since_flush += len(infos)
        if self._steps_since_flush >= self._log_freq:
            self._flush_component_means()

        return True

    def _on_training_end(self) -> None:
        """Pin the final rollout success rate into the wandb run summary.

        ``wandb`` defaults each metric's summary to its *last logged value*,
        which for the per-step ``episode/outcome_*`` fractions is just the final
        batch and reads noisily (often 0). This writes an explicit, stable
        ``episode/final_*_rate`` computed over the rolling window of the most
        recent episodes so the run summary / runs table carries a meaningful
        end-of-training number.
        """
        if not self._wandb_active() or not self._outcome_window:
            return
        n_window = len(self._outcome_window)
        for outcome in ("success", "crash", "out_of_bounds", "timeout"):
            wandb.run.summary[f"episode/final_{outcome}_rate"] = (
                self._outcome_window.count(outcome) / n_window
            )
        wandb.run.summary["episode/final_window_size"] = n_window

    def _flush_component_means(self) -> None:
        """Log running means of each reward component, then reset accumulators."""
        if self._comp_count > 0 and self._wandb_active():
            metrics = {
                f"reward/{key}": total / self._comp_count
                for key, total in self._comp_sums.items()
            }
            metrics["curriculum/task_difficulty"] = self._latest_task_difficulty
            wandb.log(metrics, step=self.num_timesteps)

        self._comp_sums.clear()
        self._comp_count = 0
        self._steps_since_flush = 0

    @staticmethod
    def _episode_reason(info: dict) -> str:
        """Recover the true terminal outcome for one finished episode.

        On auto-reset the top-level info may carry post-reset values
        ("termination_reason": "reset"); the true terminal outcome is then in
        "final_info". Prefer it so outcomes reflect the real episode end.
        """
        final_info = info.get("final_info")
        outcome_info = final_info if isinstance(final_info, dict) else info
        return str(outcome_info.get("termination_reason", "unknown"))

    def _log_episode_outcomes(self, finished: list[dict]) -> None:
        """Log outcomes/metrics aggregated over every episode that ended this step.

        With a vectorised env (``compute.n_envs > 1``) several workers routinely
        end an episode on the *same* ``num_timesteps``. Logging each one with its
        own ``wandb.log(..., step=num_timesteps)`` makes the later call overwrite
        the earlier at that shared step, so the rarer "success" one-hot is almost
        always clobbered by a co-occurring crash/out-of-bounds and reads as 0.
        Aggregating into a single log per step (outcome *fractions* + a rolling
        success rate, mean touchdown/fuel/return) keeps every outcome visible.
        """
        if not self._wandb_active():
            return

        reasons = [self._episode_reason(info) for info in finished]
        self._outcome_window.extend(reasons)
        n = len(reasons)
        n_window = len(self._outcome_window)

        metrics: dict[str, float] = {
            "episode/task_difficulty": self._latest_task_difficulty,
        }
        # Per-step fraction (keeps existing chart names) plus a rolling rate over
        # the recent window so the curve is smooth and never clobbered.
        for outcome in ("success", "crash", "out_of_bounds", "timeout"):
            metrics[f"episode/outcome_{outcome}"] = reasons.count(outcome) / n
            metrics[f"episode/{outcome}_rate"] = (
                self._outcome_window.count(outcome) / n_window
            )

        speeds: list[float] = []
        fuels: list[float] = []
        returns: list[float] = []
        lengths: list[float] = []
        for info in finished:
            terminal_obs = info.get("terminal_observation")
            if terminal_obs is None:
                terminal_obs = info.get("final_observation")
            if terminal_obs is not None:
                raw = self._scaler.unscale(np.asarray(terminal_obs, dtype=np.float64))
                speeds.append(float(np.linalg.norm(raw[3:6])))
                fuels.append(float(self._initial_fuel_kg - raw[15]))
            ep = info.get("episode")
            if ep is not None:
                returns.append(float(ep["r"]))
                lengths.append(float(ep["l"]))

        if speeds:
            metrics["episode/touchdown_speed_mps"] = float(np.mean(speeds))
        if fuels:
            metrics["episode/fuel_used_kg"] = float(np.mean(fuels))
        if returns:
            metrics["episode/return"] = float(np.mean(returns))
        if lengths:
            metrics["episode/length"] = float(np.mean(lengths))

        wandb.log(metrics, step=self.num_timesteps)
