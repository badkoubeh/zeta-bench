"""Curriculum scheduler — linear difficulty annealing.

Maps a global step count to a difficulty fraction in [0, 1], then samples
initial conditions from an envelope that widens with difficulty.

Currently the only knob the scheduler controls is the *initial conditions*
distribution. The same `progress` value is also intended to drive the
adversary weight (Phase 3) and any other config knob slated to
anneal — the env passes `progress` through to those consumers via the info
dict.

Schedule: linear lerp 0 → 1 over `cfg.env.curriculum.anneal_steps`, then
clamped at 1. Other shapes (sigmoid, step) are reserved.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf

from dynamics.types import UPRIGHT_QUAT


class Curriculum:
    """Difficulty annealing + initial-condition sampler."""

    def __init__(self, cfg: DictConfig) -> None:
        """Construct from a Hydra config exposing ``env.curriculum.*`` and
        ``env.init_conditions.*``.

        An optional ``env.curriculum.progress_override`` field pins
        :meth:`progress` to a fixed value regardless of the step argument.
        Used by evaluation scripts where ``_global_step`` is zero and would
        otherwise produce the easiest envelope.
        """
        self._anneal_steps: int = int(cfg.env.curriculum.anneal_steps)
        self._init = cfg.env.init_conditions
        override = OmegaConf.select(cfg, "env.curriculum.progress_override", default=None)
        self._progress_override: float | None = float(override) if override is not None else None

    def progress(self, step: int) -> float:
        """Linear lerp 0 → 1 over ``anneal_steps``, clamped above 1.

        Step counts beyond ``anneal_steps`` return ``1.0`` — the curriculum
        holds at full difficulty rather than going beyond. If an override
        was supplied at construction, that value is returned instead.
        """
        if self._progress_override is not None:
            return self._progress_override
        if self._anneal_steps <= 0:
            return 1.0
        return float(min(1.0, max(0.0, step / self._anneal_steps)))

    def sample_initial_conditions(
        self,
        rng: np.random.Generator,
        progress: float,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Sample initial conditions from the curriculum-lerped envelope.

        At ``progress = 0`` the envelope is at its easiest (low drop height,
        no lateral offset, slow initial descent). At ``progress = 1`` it
        widens to the full range declared in ``configs/env.yaml``.

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

        # Altitude sampled uniformly in [min, min + progress·(max-min)]
        altitude_max_at_progress = init.altitude_min_m + progress * (
            init.altitude_max_m - init.altitude_min_m
        )
        altitude = float(rng.uniform(init.altitude_min_m, altitude_max_at_progress))

        # Lateral offset: 0 at easiest, full range at hardest
        lateral_max = progress * init.lateral_offset_max_m
        x = float(rng.uniform(-lateral_max, lateral_max)) if lateral_max > 0 else 0.0
        y = float(rng.uniform(-lateral_max, lateral_max)) if lateral_max > 0 else 0.0

        # NED Z+ = down, so altitude above pad ⇒ negative z
        position_NED = np.array([x, y, -altitude], dtype=np.float64)

        # Descent velocity in [min, min + progress·(max-min)]
        vz_max_at_progress = init.descent_velocity_min_mps + progress * (
            init.descent_velocity_max_mps - init.descent_velocity_min_mps
        )
        vz = float(rng.uniform(init.descent_velocity_min_mps, vz_max_at_progress))
        velocity_NED = np.array([0.0, 0.0, vz], dtype=np.float64)

        # Attitude: upright always for now (attitude_tilt_max_rad = 0 in config)
        attitude_quat = UPRIGHT_QUAT.copy()

        # Angular rate: zero
        angular_rate_body = np.zeros(3, dtype=np.float64)

        return position_NED, velocity_NED, attitude_quat, angular_rate_body
