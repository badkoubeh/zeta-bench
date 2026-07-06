"""Control-theoretic reward function for the rocket-landing env.

The reward is the negative of a physical control cost plus a terminal landing
outcome::

    total = shaping + terminal + regularization

Potential-based progress shaping (PBRS)
---------------------------------------
The dense signal is a potential difference, NOT a raw per-step penalty::

    shaping(s, s') = gamma * Phi(s') - Phi(s)
    Phi(s)         = -landing_cost(s)

where ``landing_cost`` is a sum of *normalised squared* physical errors (lateral,
altitude, speed, vertical speed, tilt, angular rate). PBRS is policy-invariant:
the accumulated shaping over an episode telescopes to ``Phi(s_0) - gamma^T Phi(s_T)``
regardless of how long the episode runs. That removes the failure mode of the old
all-penalty reward, where a quick crash stopped accumulating per-step penalties and
therefore scored *better* than a long controlled descent. Weights come from
``configs/reward.yaml``.

Terminal reward
---------------
Encodes the true task outcome with impact-aware crash shaping so that
``safe landing > slow upright crash > fast tilted crash > out-of-bounds``::

    success       -> +success_bonus
    crash         -> -(base + k_speed·v̂² + k_tilt·t̂² + k_rate·ω̂² + k_lat·l̂²)
    out_of_bounds -> -out_of_bounds_penalty
    timeout       -> -(timeout_base + timeout_state·landing_cost(s))

All component functions return a ``(total, components)`` pair so the wandb
callback can log each term separately (per CONTRIBUTING.md §Experiment Tracking).

Tilt-from-vertical is computed *without* Euler decomposition (which has a
singularity at the upright pose): the body +X axis is rotated into the inertial
frame and the angle to inertial −Z (up in NED) is the tilt.
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


# --- landing cost / potential ---------------------------------------------

def touchdown_metrics(state: State) -> dict[str, float]:
    """Physical landing-quality metrics for a state (used by terminal shaping
    and for logging). All are non-negative magnitudes.

    Returns keys: ``speed`` (m/s), ``vertical_speed`` (m/s), ``tilt`` (rad),
    ``angular_rate`` (rad/s), ``lateral`` (m), ``altitude`` (m).
    """
    pos = position(state)
    vel = velocity(state)
    return {
        "speed": float(np.linalg.norm(vel)),
        "vertical_speed": float(abs(vel[2])),
        "tilt": tilt_from_vertical(state),
        "angular_rate": float(np.linalg.norm(angular_rate(state))),
        "lateral": float(np.linalg.norm(pos[:2])),
        "altitude": float(abs(pos[2])),
    }


def landing_cost(state: State, cfg: DictConfig) -> float:
    """Normalised physical control cost of a state — lower is more promising.

    A weighted sum of squared, scale-normalised errors. Zero only at a perfect
    landing state (on the pad, at rest, upright). Used to build the potential
    ``Phi = -landing_cost`` and the timeout terminal cost.
    """
    s = cfg.reward.scales
    p = cfg.reward.potential
    m = touchdown_metrics(state)

    lateral_n = m["lateral"] / float(s.lateral_m)
    altitude_n = m["altitude"] / float(s.altitude_m)
    speed_n = m["speed"] / float(s.velocity_mps)
    vspeed_n = m["vertical_speed"] / float(s.vertical_velocity_mps)
    tilt_n = m["tilt"] / float(s.tilt_rad)
    rate_n = m["angular_rate"] / float(s.angular_rate_radps)

    # Ground-gated terminal-velocity term: penalise speed increasingly as the
    # vehicle nears the pad, so the agent learns to flare/brake on final approach.
    # gate ∈ (0, 1], ≈1 at touchdown and decaying with altitude; zero speed (the
    # landed state) contributes nothing, keeping landing_cost zero there.
    landing_speed_weight = float(getattr(p, "landing_speed_weight", 0.0))
    gate_alt = float(getattr(p, "ground_gate_altitude_m", 15.0))
    ground_gate = float(np.exp(-m["altitude"] / gate_alt)) if gate_alt > 0.0 else 1.0

    return (
        float(p.lateral_weight) * lateral_n * lateral_n
        + float(p.altitude_weight) * altitude_n * altitude_n
        + float(p.velocity_weight) * speed_n * speed_n
        + float(p.vertical_velocity_weight) * vspeed_n * vspeed_n
        + float(p.tilt_weight) * tilt_n * tilt_n
        + float(p.angular_rate_weight) * rate_n * rate_n
        + landing_speed_weight * speed_n * speed_n * ground_gate
    )


def potential(state: State, cfg: DictConfig) -> float:
    """Shaping potential ``Phi(state) = -landing_cost(state)``, clipped to
    ``reward.potential.clip_abs`` for numerical stability."""
    clip = float(cfg.reward.potential.clip_abs)
    return float(np.clip(-landing_cost(state, cfg), -clip, clip))


# --- dense (potential-based) shaping + regularization ---------------------

def shaping_reward(
    prev_state: State,
    next_state: State | None,
    action: Action,
    prev_action: Action | None,
    prev_fuel_mass: float,
    gamma: float,
    cfg: DictConfig,
) -> tuple[float, dict[str, float]]:
    """Per-step dense reward: potential-based shaping + small regularizers.

    Parameters
    ----------
    prev_state : State
        14-dim state BEFORE the dynamics step.
    next_state : State | None
        14-dim state AFTER the step, or ``None`` to signal a terminal
        transition. For terminal transitions the next-state potential is taken
        as 0 (standard PBRS absorbing-state convention), so the episode's
        accumulated shaping telescopes to ``-Phi(s_0)`` and the outcome is
        carried entirely by :func:`terminal_reward`.
    action : Action
        3-dim action just executed.
    prev_action : Action | None
        Previous step's action, or ``None`` on the first step.
    prev_fuel_mass : float
        Fuel mass (kg) BEFORE the step — used to compute fuel burned.
    gamma : float
        RL discount factor. Must match the agent's gamma for PBRS invariance.
    cfg : DictConfig
        Composed Hydra config exposing ``reward.*``.

    Returns
    -------
    tuple[float, dict[str, float]]
        ``(total, components)`` with keys ``shaping``, ``fuel``, ``smoothness``,
        ``control`` (one per logged term).
    """
    r = cfg.reward.regularization

    phi_prev = potential(prev_state, cfg)
    phi_next = 0.0 if next_state is None else potential(next_state, cfg)
    shaping = gamma * phi_next - phi_prev

    next_fuel = prev_fuel_mass if next_state is None else fuel_mass(next_state)
    fuel_burned = max(0.0, prev_fuel_mass - next_fuel)
    fuel_pen = -float(r.fuel_weight) * fuel_burned

    if prev_action is None:
        smoothness_pen = 0.0
    else:
        diff = action - prev_action
        smoothness_pen = -float(r.smoothness_weight) * float(np.dot(diff, diff))

    control_pen = -float(r.control_weight) * float(np.dot(action, action))

    components: dict[str, float] = {
        "shaping": shaping,
        "fuel": fuel_pen,
        "smoothness": smoothness_pen,
        "control": control_pen,
    }
    total = sum(components.values())
    return total, components


# --- terminal outcome ------------------------------------------------------

def terminal_reward(reason: str, state: State, cfg: DictConfig) -> float:
    """Impact-aware terminal reward for the given termination reason.

    Returns 0.0 for non-terminal reasons (``"ongoing"``, ``"reset"``) so the
    caller can blindly add it. ``"timeout"`` carries a state-quality cost.

    Recognised reasons: ``"success"``, ``"crash"``, ``"out_of_bounds"``,
    ``"timeout"``.
    """
    t = cfg.reward.terminal
    s = cfg.reward.scales
    clip = float(t.terminal_clip_abs)

    if reason == "success":
        return float(np.clip(float(t.success_bonus), -clip, clip))

    if reason == "crash":
        m = touchdown_metrics(state)
        speed_n = m["speed"] / float(s.velocity_mps)
        tilt_n = m["tilt"] / float(s.tilt_rad)
        rate_n = m["angular_rate"] / float(s.angular_rate_radps)
        lateral_n = m["lateral"] / float(s.lateral_m)
        penalty = (
            float(t.crash_base_penalty)
            + float(t.touchdown_speed_weight) * speed_n * speed_n
            + float(t.touchdown_tilt_weight) * tilt_n * tilt_n
            + float(t.touchdown_angular_rate_weight) * rate_n * rate_n
            + float(t.touchdown_lateral_weight) * lateral_n * lateral_n
        )
        return float(np.clip(-penalty, -clip, clip))

    if reason == "out_of_bounds":
        return float(np.clip(-float(t.out_of_bounds_penalty), -clip, clip))

    if reason == "timeout":
        penalty = float(t.timeout_base_penalty) + float(
            t.timeout_state_weight
        ) * landing_cost(state, cfg)
        return float(np.clip(-penalty, -clip, clip))

    return 0.0
