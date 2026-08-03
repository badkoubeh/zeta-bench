"""Model-predictive control baseline — convex receding-horizon descent guidance.

What this is
------------
A **vertical-channel** guidance law. Every ``resolve_every_n_ticks`` control
ticks it solves a finite-horizon optimal-control problem for the throttle
sequence that best tracks a physically-derived deceleration envelope, applies
the first move, and re-plans. Gimbal is commanded zero, exactly matching the
PID baseline's active scope (``controllers/pid_baseline.py``) so the two are
compared on equal terms — no controller in the current task performs lateral
guidance.

What this is **not**
--------------------
This is not the 3-DOF lossless-convexified second-order-cone program that
operators actually fly (Açıkmeşe & Ploen, *Convex Programming Approach to
Powered Descent Guidance for Mars Landing*, JGCD 30(5), 2007; Blackmore et al.
on the G-FOLD lineage). It is analogous in *structure* — relax a non-convex
input set, solve a convex program every cycle, apply the first move — but it
plans one axis with a quadratic cost, not three axes with cone constraints.
Read the robustness card as evidence about *this* law, not about SOCP guidance.

Why it should differ from the PID baseline
------------------------------------------
Both controllers ultimately regulate descent rate, but the reference differs in
kind. PID follows a fixed linear flare (``0.10 · altitude``, tuned offline).
The MPC recomputes its envelope every solve from the *measured* mass, tilt, and
thrust authority, so a heavier vehicle or a spent tank automatically produces a
more conservative profile without anyone re-tuning a gain. That is the property
the mass-offset axis of the disturbance matrix is built to probe.

The internal model
------------------
State ``x = [z, vz]`` in NED (z down-positive, so altitude ``h = −z``; vz
positive descending). Over one horizon step ``Δ``::

    ż  = vz
    v̇z = g − (u · T_max · cosθ)/m − (b/m)·vz + c·|vz₀|·vz₀/m

Three linearizations make this LTI over the horizon. Each is a deliberate,
bounded approximation, not an oversight:

============  =========================================  ==========================================
Term          Treatment                                  Why it is acceptable
============  =========================================  ==========================================
mass ``m``    frozen at the measured total mass          ≤1.4% drift over a 3 s horizon on a 30 t
                                                         vehicle
drag          Jacobian linearization about ``vz₀``:      exact value *and* slope at the current
              ``F ≈ −2c|vz₀|·vz + c|vz₀|·vz₀``          operating point, which is where the plan
                                                         starts
tilt ``cosθ`` frozen at the measured value               the controller does not command attitude,
                                                         so tilt is exogenous
============  =========================================  ==========================================

Physical constants and the vehicle parameters come from
:mod:`dynamics.equations_of_motion` and ``cfg.env.dynamics`` respectively — never
re-declared here, so the controller's model cannot silently drift away from the
plant it is steering.

**The model is nominal.** It is built from the config's nominal parameters and
sees the world only through the observation. It has no knowledge of the wind,
mass offset, or sensor noise injected by a disturbance cell, so under those
cells it is planning with a *wrong* model. That is deliberate: it is the same
information the PID baseline gets, and it is what makes the disturbance matrix
a test of robustness to model mismatch rather than a test of privileged access.

The non-convex input set
------------------------
``dynamics.equations_of_motion.clamp_throttle`` makes the achievable thrust set
disjoint — a command ``≤ 0`` shuts the engine off, anything above is clipped up
to ``throttle_min`` — so the admissible set is ``{0} ∪ [throttle_min, 1]``. No
quadratic program can express that. The solver therefore plans over the convex
hull ``[0, 1]`` and :meth:`_project_to_plant_input` maps the applied first move
back onto the real set. The relaxation is *not* proven lossless here (unlike
LCvx), so the projection introduces genuine plan-vs-plant mismatch; it is
smallest where it matters most, since hover throttle (≈0.35) sits comfortably
above the ``0.25`` floor and the terminal burn never enters the dead zone.

State (carried across ``predict()`` calls; cleared by :meth:`reset`):
    - the current plan (throttle sequence) and ticks since it was solved
    - the last applied throttle, which anchors the Δu smoothing term
    - solve-time telemetry, reported by :meth:`solve_stats`
"""
from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig
from scipy.optimize import lsq_linear

from dynamics.equations_of_motion import G_EARTH, RHO_AIR_SEA_LEVEL
from utils.normalisation import FixedObsScaler

# Below this damping coefficient the exact-discretization expressions
# (1−e^{−aΔ})/a and (Δ−(1−e^{−aΔ})/a)/a lose precision to cancellation; the
# a→0 limits (Δ and Δ²/2) are used instead. Numerical guard, not a physical
# parameter.
_DAMPING_EPS: float = 1e-9

# Floor on cos(tilt), the fraction of thrust that acts along the vertical. Keeps
# the thrust-authority term positive (and the prediction matrices full rank)
# when a tumbling vehicle or a sensor spike reports a near-horizontal nose.
# 0.1 ⇒ tilt of ~84°, far past any recoverable attitude.
_MIN_COS_TILT: float = 0.1


class MPCController:
    """Receding-horizon vertical-descent guidance. Consumes the env's 17-dim
    scaled observation (the constructor takes a config with both
    ``env.obs_scaler`` bounds and ``mpc_controller`` settings so the controller
    can unscale obs to physical units internally and then act).
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Construct from a Hydra config exposing ``mpc_controller``,
        ``env.dynamics``, ``env.obs_scaler``, and ``env.episode.control_hz``."""
        self._cfg = cfg
        mpc = cfg.mpc_controller

        # --- horizon / cadence ---
        self._horizon = int(mpc.horizon)
        self._step_ticks = int(mpc.step_ticks)
        self._resolve_every = int(mpc.resolve_every_n_ticks)
        if self._horizon < 1 or self._step_ticks < 1 or self._resolve_every < 1:
            raise ValueError(
                "mpc_controller.horizon, .step_ticks and .resolve_every_n_ticks "
                "must all be >= 1 (got "
                f"{self._horizon}, {self._step_ticks}, {self._resolve_every})"
            )
        if self._resolve_every > self._step_ticks * self._horizon:
            raise ValueError(
                f"mpc_controller.resolve_every_n_ticks={self._resolve_every} exceeds the "
                f"plan's own span ({self._step_ticks} × {self._horizon} ticks) — the plan "
                "would run out before the next solve"
            )
        # Horizon step: the optimiser discretisation, coarser than the control tick.
        self._delta = self._step_ticks / float(cfg.env.episode.control_hz)

        # --- nominal plant parameters (NOT the disturbed ones) ---
        d = cfg.env.dynamics
        self._dry_mass_kg = float(d.dry_mass_kg)
        self._max_thrust_N = float(d.max_thrust_N)
        self._throttle_min = float(d.throttle_min)
        self._throttle_max = float(d.throttle_max)
        # Quadratic-drag coefficient c in F_drag = −c·|v|·v.
        self._drag_c = (
            0.5 * RHO_AIR_SEA_LEVEL * float(d.drag_coefficient) * float(d.reference_area_m2)
        )

        # --- deceleration envelope ---
        env = mpc.envelope
        self._touchdown_target_mps = float(env.touchdown_target_mps)
        self._envelope_margin = float(env.margin)
        self._min_descent_mps = float(env.min_descent_mps)
        self._max_descent_mps = float(env.max_descent_mps)

        # --- cost weights (stored as sqrt: the LSQ stacks residuals, not the cost) ---
        w = mpc.weights
        self._sqrt_w_velocity = float(np.sqrt(float(w.velocity)))
        self._sqrt_w_throttle = float(np.sqrt(float(w.throttle)))
        self._sqrt_w_throttle_rate = float(np.sqrt(float(w.throttle_rate)))

        self._ignition_threshold = float(mpc.ignition_threshold)

        # Need to unscale obs to physical units before acting
        self._scaler = FixedObsScaler(cfg)

        # Plan / warm-start state
        self._plan: NDArray[np.float64] | None = None
        self._ticks_since_solve: int = 0
        self._last_throttle: float = 0.0
        self._solve_times_s: list[float] = []

    # --- episode lifecycle ---------------------------------------------------

    def reset(self) -> None:
        """Discard the plan, warm start, and solve telemetry at episode start."""
        self._plan = None
        self._ticks_since_solve = 0
        self._last_throttle = 0.0
        self._solve_times_s = []

    def solve_stats(self) -> dict[str, float]:
        """Solve-time telemetry for the episodes since the last :meth:`reset`.

        Reported by the eval entrypoints so the optimiser's cost is a measured
        number rather than an assumption — the full disturbance matrix is
        millions of control ticks, and this is what says whether the configured
        cadence is affordable.
        """
        if not self._solve_times_s:
            return {"n_solves": 0, "solve_ms_mean": 0.0, "solve_ms_p99": 0.0}
        times_ms = np.asarray(self._solve_times_s, dtype=np.float64) * 1e3
        return {
            "n_solves": float(times_ms.size),
            "solve_ms_mean": float(times_ms.mean()),
            "solve_ms_p99": float(np.percentile(times_ms, 99)),
        }

    # --- the control law -----------------------------------------------------

    def predict(
        self,
        obs: NDArray[np.float64],
        deterministic: bool = True,
    ) -> NDArray[np.float64]:
        """Compute a 3-dim action from a 17-dim scaled observation.

        Re-solves the horizon problem when the plan is stale, otherwise applies
        the queued move (receding horizon with a configurable replan interval).
        The ``deterministic`` flag is accepted for interface parity with the SB3
        agents; this controller has no stochastic component, so it is ignored —
        an identical observation sequence always yields an identical action
        sequence, which is what the fixed-seed matrix relies on.
        """
        raw = self._scaler.unscale(obs)
        altitude_m = max(0.0, -float(raw[2]))
        descent_rate_mps = float(raw[5])
        # cos(tilt-from-vertical) is exactly sin(pitch): the nose-up projection
        # 2(wy − xz) that envs.reward.tilt_from_vertical takes the arccos of is
        # the same quantity quat_to_euler feeds to arcsin for pitch. Recovering
        # it this way needs no quaternion and has no gimbal-lock caveat.
        cos_tilt = float(np.clip(np.sin(float(raw[7])), _MIN_COS_TILT, 1.0))
        total_mass_kg = self._dry_mass_kg + max(0.0, float(raw[15]))

        if self._plan is None or self._ticks_since_solve >= self._resolve_every:
            self._plan = self._solve(altitude_m, descent_rate_mps, total_mass_kg, cos_tilt)
            self._ticks_since_solve = 0

        step_index = min(self._ticks_since_solve // self._step_ticks, self._horizon - 1)
        self._ticks_since_solve += 1

        throttle = self._project_to_plant_input(float(self._plan[step_index]))
        self._last_throttle = throttle

        # Vertical channel only — no lateral guidance, matching the PID baseline.
        return np.array([throttle, 0.0, 0.0], dtype=np.float64)

    # --- pieces of the control law (separated so each is directly testable) --

    def descent_envelope(
        self,
        altitude_m: NDArray[np.float64] | float,
        total_mass_kg: float,
        cos_tilt: float,
    ) -> NDArray[np.float64]:
        """Maximum descent rate that is still stoppable at each altitude.

        From constant-acceleration kinematics, a vehicle able to decelerate at
        ``a`` can shed speed to ``v_td`` over height ``h`` iff::

            v(h) = sqrt(v_td² + 2·a·h)

        with ``a = margin · (T_max·cosθ/m − g)`` — a *fraction* of the currently
        available deceleration, so the envelope keeps braking authority in
        reserve for whatever the disturbance is doing. Because ``a`` is
        recomputed from the measured mass and tilt on every solve, the envelope
        tightens by itself as the vehicle gets heavier or the tank empties;
        this is the substantive difference from a fixed tuned flare.

        Clipped to ``[min_descent_mps, max_descent_mps]``: the floor keeps the
        vehicle descending rather than hovering out the clock, the cap keeps the
        reference inside the modelled descent-speed range.
        """
        thrust_authority = self._max_thrust_N * cos_tilt / total_mass_kg
        # A vehicle that cannot out-thrust gravity has no braking authority; the
        # tiny floor keeps the square root real and the envelope monotone.
        decel = max(self._envelope_margin * (thrust_authority - G_EARTH), _DAMPING_EPS)
        h = np.maximum(np.asarray(altitude_m, dtype=np.float64), 0.0)
        envelope = np.sqrt(self._touchdown_target_mps**2 + 2.0 * decel * h)
        return np.clip(envelope, self._min_descent_mps, self._max_descent_mps)

    def build_prediction_matrices(
        self,
        altitude_m: float,
        descent_rate_mps: float,
        total_mass_kg: float,
        cos_tilt: float,
    ) -> tuple[
        NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
    ]:
        """Condense the horizon model into altitude/velocity profiles affine in ``u``.

        Returns ``(S_z, f_z, S_v, f_v)`` such that, for a throttle sequence
        ``u`` of length ``horizon``, the predicted profiles over steps
        ``1..horizon`` are ``z = S_z @ u + f_z`` and ``vz = S_v @ u + f_v``.
        Both ``S`` matrices are lower triangular — step *k* cannot depend on a
        throttle applied after it.

        The per-step map is the *exact* zero-order-hold discretization of the
        linearized model (not a forward-Euler approximation), so accuracy does
        not degrade as ``step_ticks`` is raised to buy a longer horizon::

            v_{k+1} = E·v_k + Φ·(g_eff − β·u_k)
            z_{k+1} = z_k + Φ·v_k + Ψ·(g_eff − β·u_k)

        with ``E = e^{−aΔ}``, ``Φ = (1−E)/a``, ``Ψ = (Δ−Φ)/a``, damping
        ``a = 2c|vz₀|/m``, thrust authority ``β = T_max·cosθ/m``, and the
        drag linearization's constant part folded into
        ``g_eff = g + c|vz₀|vz₀/m``.
        """
        n = self._horizon
        dt = self._delta

        speed = abs(descent_rate_mps)
        damping = 2.0 * self._drag_c * speed / total_mass_kg
        thrust_per_throttle = self._max_thrust_N * cos_tilt / total_mass_kg
        gravity_eff = G_EARTH + self._drag_c * speed * descent_rate_mps / total_mass_kg

        if damping > _DAMPING_EPS:
            decay = float(np.exp(-damping * dt))
            phi = (1.0 - decay) / damping
            psi = (dt - phi) / damping
        else:  # a → 0 limits (no measurable drag, e.g. at rest)
            decay = 1.0
            phi = dt
            psi = 0.5 * dt * dt

        # x_{k+1} = A x_k + B u_k + c_vec, with x = [z, vz] (NED, z down-positive).
        A = np.array([[1.0, phi], [0.0, decay]], dtype=np.float64)
        B = np.array([-psi * thrust_per_throttle, -phi * thrust_per_throttle], dtype=np.float64)
        c_vec = np.array([psi * gravity_eff, phi * gravity_eff], dtype=np.float64)

        # Roll the recursion forward, carrying x_k as an affine function of u:
        # a constant part F (shape (2,)) and u-coefficients G (shape (2, n)).
        S_z = np.zeros((n, n), dtype=np.float64)
        S_v = np.zeros((n, n), dtype=np.float64)
        f_z = np.zeros(n, dtype=np.float64)
        f_v = np.zeros(n, dtype=np.float64)
        F = np.array([-altitude_m, descent_rate_mps], dtype=np.float64)  # z₀ = −h₀
        G = np.zeros((2, n), dtype=np.float64)
        for k in range(n):
            F = A @ F + c_vec
            G = A @ G
            G[:, k] += B
            f_z[k], f_v[k] = F
            S_z[k] = G[0]
            S_v[k] = G[1]

        return S_z, f_z, S_v, f_v

    def _solve(
        self,
        altitude_m: float,
        descent_rate_mps: float,
        total_mass_kg: float,
        cos_tilt: float,
    ) -> NDArray[np.float64]:
        """Solve one horizon problem; return the throttle sequence.

        The cost is quadratic and the only constraints are the throttle box, so
        the condensed problem is a *bounded-variable least squares* — the three
        weighted residual blocks are stacked into one system and handed to
        ``scipy.optimize.lsq_linear`` with the BVLS active-set method, which
        terminates at the exact optimum rather than at a tolerance::

            minimise ‖√w_v (S_v u + f_v − v_env)‖² + ‖√w_u u‖² + ‖√w_du (D u − d₀)‖²
            over     0 ≤ u ≤ 1

        The envelope reference ``v_env`` depends on the predicted altitude,
        which depends on ``u`` — so it is evaluated along the *previous*
        solution's trajectory (shifted one step, the standard warm start). That
        is one sequential-convex-programming pass per control cycle, not an
        iteration to convergence: replanning every few ticks makes further
        passes redundant.
        """
        started = time.perf_counter()
        n = self._horizon
        S_z, f_z, S_v, f_v = self.build_prediction_matrices(
            altitude_m, descent_rate_mps, total_mass_kg, cos_tilt
        )

        # Warm start: last plan shifted forward, else a constant hover estimate.
        if self._plan is None:
            thrust_authority_N = self._max_thrust_N * cos_tilt
            hover_throttle = np.clip(G_EARTH * total_mass_kg / thrust_authority_N, 0.0, 1.0)
            u_warm = np.full(n, hover_throttle)
        else:
            u_warm = np.roll(self._plan, -1)
            u_warm[-1] = self._plan[-1]

        predicted_altitude = np.maximum(-(S_z @ u_warm + f_z), 0.0)
        v_env = self.descent_envelope(predicted_altitude, total_mass_kg, cos_tilt)

        # Residual block 1: envelope tracking. S_v is lower triangular with a
        # non-zero diagonal, so this block alone makes the stack full rank —
        # the regularisers below may safely be switched off by zero weights.
        blocks = [self._sqrt_w_velocity * S_v]
        targets = [self._sqrt_w_velocity * (v_env - f_v)]

        # Block 2: throttle magnitude (fuel proxy).
        if self._sqrt_w_throttle > 0.0:
            blocks.append(self._sqrt_w_throttle * np.eye(n))
            targets.append(np.zeros(n))

        # Block 3: throttle rate, anchored to the throttle actually applied last
        # tick so the plan cannot open with a step the actuator never made.
        if self._sqrt_w_throttle_rate > 0.0:
            difference = np.eye(n) - np.eye(n, k=-1)
            anchor = np.zeros(n)
            anchor[0] = self._last_throttle
            blocks.append(self._sqrt_w_throttle_rate * difference)
            targets.append(self._sqrt_w_throttle_rate * anchor)

        solution = lsq_linear(
            np.vstack(blocks),
            np.concatenate(targets),
            bounds=(0.0, 1.0),
            method="bvls",
        )
        self._solve_times_s.append(time.perf_counter() - started)
        return np.asarray(solution.x, dtype=np.float64)

    def _project_to_plant_input(self, throttle: float) -> float:
        """Map a relaxed throttle onto the plant's disjoint admissible set.

        The optimiser plans over ``[0, 1]``, but the plant can only deliver zero
        thrust or thrust at ``throttle_min`` and above (see
        ``dynamics.equations_of_motion.clamp_throttle``). Commands below
        ``ignition_threshold`` are read as "the plan wants less than the engine
        can give" and shut it off; the rest are raised to the floor. Sending the
        already-projected value keeps the observation's ``last_action`` channel
        equal to the true actuator state.
        """
        if throttle < self._ignition_threshold:
            return 0.0
        return float(np.clip(throttle, self._throttle_min, self._throttle_max))

    # --- persistence ---------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist settings to disk (YAML)."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "MPCController":
        """Restore controller from saved settings."""
        raise NotImplementedError
