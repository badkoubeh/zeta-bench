"""Hybrid reward function for the rocket-landing env.

Dense shaping (per step) + sparse terminal (episode end). Weights come from
``configs/reward.yaml``. The function returns a *component breakdown* alongside
the total so the wandb callback can log each term separately (per CONTRIBUTING.md
§Experiment Tracking).

Dense terms (each a *penalty*, so negative-valued):
    - distance:   ‖position‖₂                          (m)        × distance_weight
    - velocity:   ‖velocity‖₂                          (m/s)      × velocity_weight
    - attitude:   tilt-from-vertical                   (rad)      × attitude_weight
    - angular_rate: ‖ω‖₂                               (rad/s)    × angular_rate_weight
    - fuel:       fuel-burned-this-step                (kg)       × fuel_weight
    - smoothness: ‖action − prev_action‖₂²             (-)        × smoothness_weight

Sparse terminal (signed; sign chosen by config):
    - success_bonus      on safe touchdown    (positive)
    - crash_penalty      on hard touchdown    (negative)
    - out_of_bounds_penalty on OOB cylinder   (negative)

Tilt-from-vertical is computed *without* going through Euler decomposition
(which has a singularity at the perfectly-upright pose). The body +X axis is
rotated into the inertial frame; the angle between that and the inertial
−Z axis (up direction in NED) is the tilt.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from dynamics.equations_of_motion import quat_rotate_body_to_inertial
from dynamics.types import (
    Action,
    State,
    angular_rate,
    fuel_mass,
    position,
    quaternion,
    velocity,
)

_UP_INERTIAL: NDArray[np.float64] = np.array([0.0, 0.0, -1.0], dtype=np.float64)
_BODY_NOSE: NDArray[np.float64] = np.array([1.0, 0.0, 0.0], dtype=np.float64)


def tilt_from_vertical(state: State) -> float:
    """Angle (rad) between the rocket's nose (body +X) and the inertial up
    direction (−Z in NED). Zero when perfectly upright.

    Uses a direct quaternion→vector projection rather than Euler angles to
    avoid the gimbal-lock singularity at pitch = ±π/2 (which is the nose-up
    pose for our tail-firing rocket).
    """
    q = quaternion(state)
    nose_in_inertial = quat_rotate_body_to_inertial(q, _BODY_NOSE)
    cos_tilt = float(np.clip(np.dot(nose_in_inertial, _UP_INERTIAL), -1.0, 1.0))
    return float(np.arccos(cos_tilt))


def dense_reward(
    state: State,
    action: Action,
    prev_action: Action | None,
    prev_fuel_mass: float,
    cfg: DictConfig,
) -> tuple[float, dict[str, float]]:
    """Compute the per-step dense reward + component breakdown.

    All terms are *signed*: positive when behaviour is desirable, negative
    when undesirable. With the default config every dense term is a penalty
    (negative-valued), reflecting the standard shaping convention "subtract
    cost from value".

    Parameters
    ----------
    state : State
        14-dim state vector AFTER the dynamics step.
    action : Action
        3-dim action just executed.
    prev_action : Action | None
        Previous step's action, or ``None`` on the first step.
    prev_fuel_mass : float
        Fuel mass (kg) BEFORE the step — used to compute fuel burned.
    cfg : DictConfig
        Composed Hydra config exposing ``reward.dense.*`` weights.

    Returns
    -------
    tuple[float, dict[str, float]]
        ``(total_dense_reward, components_dict)`` where ``components_dict``
        has one entry per dense term plus a sentinel for the eventual
        terminal contribution (set to 0 here — terminal added separately
        by the env on episode end).
    """
    w = cfg.reward.dense

    distance_pen = -w.distance_weight * float(np.linalg.norm(position(state)))
    velocity_pen = -w.velocity_weight * float(np.linalg.norm(velocity(state)))
    attitude_pen = -w.attitude_weight * tilt_from_vertical(state)
    angular_rate_pen = -w.angular_rate_weight * float(
        np.linalg.norm(angular_rate(state))
    )

    fuel_burned = max(0.0, prev_fuel_mass - fuel_mass(state))
    fuel_pen = -w.fuel_weight * fuel_burned

    if prev_action is None:
        smoothness_pen = 0.0
    else:
        diff = action - prev_action
        smoothness_pen = -w.smoothness_weight * float(np.dot(diff, diff))

    components: dict[str, float] = {
        "distance": distance_pen,
        "velocity": velocity_pen,
        "attitude": attitude_pen,
        "angular_rate": angular_rate_pen,
        "fuel": fuel_pen,
        "smoothness": smoothness_pen,
    }
    total = sum(components.values())
    return total, components


def terminal_reward(reason: str, cfg: DictConfig) -> float:
    """Look up the sparse terminal reward for the given termination reason.

    Returns 0.0 for unrecognised reasons (e.g. "ongoing", "timeout") so the
    caller can blindly add it to the dense reward without special-casing.

    Recognised reasons: ``"success"``, ``"crash"``, ``"out_of_bounds"``.
    """
    s = cfg.reward.sparse
    if reason == "success":
        return float(s.success_bonus)
    if reason == "crash":
        return float(s.crash_penalty)
    if reason == "out_of_bounds":
        return float(s.out_of_bounds_penalty)
    return 0.0
