"""Curriculum scheduler — linear task-difficulty annealing.

Maps a global step count to a *task difficulty* fraction in [0, 1], then
samples initial conditions from an envelope that widens with difficulty.

``task_difficulty`` is the hardness of the *nominal* landing problem — the
initial-condition envelope (drop height, lateral offset, descent speed,
attitude tilt). It is the training curriculum scalar **only**: it does NOT
scale environmental disturbances or adversary weight. Those belong to the
separate ``disturbance_severity`` (graduated matrix) and adversarial axes (see
the "Naming conventions" section in ``CONTRIBUTING.md``); keeping them decoupled
is what keeps the training curriculum and the graduated disturbance matrix
independent.

Schedule (``cfg.env.curriculum.schedule``):
- ``linear`` — lerp 0 → 1 over ``cfg.env.curriculum.anneal_steps``, then
  clamped at 1 (the default training schedule).
- ``fixed``  — hold at ``cfg.env.curriculum.task_difficulty`` regardless of the
  step count. Used by evaluation, where ``_global_step`` is zero and a linear
  schedule would otherwise produce the easiest envelope.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf

from dynamics.types import UPRIGHT_QUAT


class Curriculum:
    """Task-difficulty annealing + initial-condition sampler."""

    def __init__(self, cfg: DictConfig) -> None:
        """Construct from a Hydra config exposing ``env.curriculum.*`` and
        ``env.init_conditions.*``.

        ``env.curriculum.schedule`` selects the annealing mode (``linear`` or
        ``fixed``). Under ``fixed`` the scheduler pins :meth:`task_difficulty`
        to ``env.curriculum.task_difficulty`` regardless of the step argument —
        used by evaluation scripts where ``_global_step`` is zero and a linear
        schedule would otherwise produce the easiest envelope.
        """
        self._anneal_steps: int = int(cfg.env.curriculum.anneal_steps)
        self._init = cfg.env.init_conditions
        self._schedule: str = str(
            OmegaConf.select(cfg, "env.curriculum.schedule", default="linear")
        )
        if self._schedule not in ("linear", "fixed"):
            raise ValueError(
                "env.curriculum.schedule must be 'linear' or 'fixed', "
                f"got {self._schedule!r}"
            )
        self._fixed_task_difficulty: float = float(
            OmegaConf.select(cfg, "env.curriculum.task_difficulty", default=1.0)
        )

    def task_difficulty(self, step: int) -> float:
        """Current task difficulty in [0, 1] for the given global step.

        Under the ``fixed`` schedule the configured
        ``env.curriculum.task_difficulty`` is returned regardless of ``step``.
        Under ``linear`` the value lerps 0 → 1 over ``anneal_steps`` and is
        clamped above 1 — steps beyond ``anneal_steps`` hold at full difficulty
        rather than going beyond. A non-positive ``anneal_steps`` also yields
        ``1.0`` (degenerate "always hardest" schedule).
        """
        if self._schedule == "fixed":
            return self._fixed_task_difficulty
        if self._anneal_steps <= 0:
            return 1.0
        return float(min(1.0, max(0.0, step / self._anneal_steps)))

    def sample_initial_conditions(
        self,
        rng: np.random.Generator,
        task_difficulty: float,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Sample initial conditions from the difficulty-lerped envelope.

        At ``task_difficulty = 0`` the envelope is at its easiest (low drop
        height, no lateral offset, slow initial descent). At
        ``task_difficulty = 1`` it widens to the full range declared in
        ``configs/env.yaml``.

        Returns
        -------
        tuple
            ``(position_NED, velocity_NED, attitude_quat, angular_rate_body)``.
            Position is above the pad (NED z < 0); velocity is descending
            (NED vz > 0); attitude is nose-up (UPRIGHT_QUAT) until the
            ``attitude_tilt_max_rad`` config knob is raised above zero in a
            future curriculum extension; angular rate starts at zero.
        """
        init = self._init

        # Altitude sampled uniformly in [min, min + difficulty·(max-min)]
        altitude_max_at_difficulty = init.altitude_min_m + task_difficulty * (
            init.altitude_max_m - init.altitude_min_m
        )
        altitude = float(rng.uniform(init.altitude_min_m, altitude_max_at_difficulty))

        # Lateral offset: 0 at easiest, full range at hardest
        lateral_max = task_difficulty * init.lateral_offset_max_m
        x = float(rng.uniform(-lateral_max, lateral_max)) if lateral_max > 0 else 0.0
        y = float(rng.uniform(-lateral_max, lateral_max)) if lateral_max > 0 else 0.0

        # NED Z+ = down, so altitude above pad ⇒ negative z
        position_NED = np.array([x, y, -altitude], dtype=np.float64)

        # Descent velocity in [min, min + difficulty·(max-min)]
        vz_max_at_difficulty = init.descent_velocity_min_mps + task_difficulty * (
            init.descent_velocity_max_mps - init.descent_velocity_min_mps
        )
        vz = float(rng.uniform(init.descent_velocity_min_mps, vz_max_at_difficulty))
        velocity_NED = np.array([0.0, 0.0, vz], dtype=np.float64)

        # Attitude: upright always for now (attitude_tilt_max_rad = 0 in config)
        attitude_quat = UPRIGHT_QUAT.copy()

        # Angular rate: zero
        angular_rate_body = np.zeros(3, dtype=np.float64)

        return position_NED, velocity_NED, attitude_quat, angular_rate_body
