"""Controllers: PID baseline, MPC baseline, SAC agent, PPO agent.

All controllers share a uniform evaluation interface (``predict``, ``save``,
``load``) so the robustness evaluation script can drive them through the
same harness regardless of underlying algorithm.

Import rule: this package may import from ``envs/``, ``dynamics/`` (read-only
types), and ``utils/``. It may be imported by ``experiments/``.
"""
from __future__ import annotations

from controllers.mpc_baseline import MPCController
from controllers.pid_baseline import PIDController
from controllers.ppo_agent import PPOAgent
from controllers.sac_agent import SACAgent

__all__ = ["MPCController", "PIDController", "PPOAgent", "SACAgent"]
