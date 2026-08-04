"""Unit tests for :class:`controllers.mpc_baseline.MPCController`.

Four groups:

- **Interface parity** with the PID baseline, so the robustness harness can
  drive both through the same loop.
- **Prediction-model correctness** — the condensed horizon matrices checked
  against an independent fine-step integration of the same continuous model.
  This is where model-predictive implementations actually go wrong, so it is
  verified numerically rather than by re-deriving the algebra in the assertion.
- **Physical sanity** of the resulting commands (coast / brake / balance).
- **Determinism and replan cadence**, the invariants the fixed-seed matrix
  depends on.
"""
from __future__ import annotations

import numpy as np
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from controllers.mpc_baseline import MPCController
from dynamics.equations_of_motion import G_EARTH, RHO_AIR_SEA_LEVEL
from envs.rocket_landing_env import OBS_DIM, RocketLandingEnv
from utils.normalisation import FixedObsScaler

# Nose-up: body +X along inertial −Z. cos(tilt) = sin(pitch) = 1 here.
_UPRIGHT_PITCH_RAD = np.pi / 2


@pytest.fixture(autouse=True)
def _clear_hydra():
    """Reset Hydra's global singleton around each test (compose hygiene)."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    yield
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


def _compose(overrides: list[str] | None = None):
    with initialize(config_path="../configs", version_base=None):
        return compose(config_name="eval_mpc", overrides=overrides or [])


@pytest.fixture
def cfg():
    return _compose()


def _make_obs(
    cfg,
    altitude_m: float,
    descent_rate_mps: float,
    fuel_kg: float | None = None,
    pitch_rad: float = _UPRIGHT_PITCH_RAD,
) -> np.ndarray:
    """Build a scaled 17-dim observation from physical quantities.

    Mirrors ``RocketLandingEnv._build_obs``: altitude is NED z = −h, descent is
    positive vz, pitch sits at index 7, fuel mass at index 15.
    """
    scaler = FixedObsScaler(cfg)
    raw = np.zeros(OBS_DIM, dtype=np.float64)
    raw[2] = -altitude_m
    raw[5] = descent_rate_mps
    raw[7] = pitch_rad
    raw[15] = float(cfg.env.dynamics.initial_fuel_kg) if fuel_kg is None else fuel_kg
    return scaler.scale(raw)


def _total_mass(cfg, fuel_kg: float | None = None) -> float:
    fuel = float(cfg.env.dynamics.initial_fuel_kg) if fuel_kg is None else fuel_kg
    return float(cfg.env.dynamics.dry_mass_kg) + fuel


def _hover_throttle(cfg) -> float:
    """Throttle that exactly cancels weight for an upright, full vehicle."""
    return _total_mass(cfg) * G_EARTH / float(cfg.env.dynamics.max_thrust_N)


class TestInterfaceParity:
    def test_predict_returns_3dim_action(self, cfg) -> None:
        """predict() returns a 3-vector regardless of input observation."""
        mpc = MPCController(cfg)
        action = mpc.predict(np.zeros(OBS_DIM))
        assert action.shape == (3,)
        assert action.dtype == np.float64

    def test_action_in_env_action_space(self, cfg) -> None:
        """Whatever MPC outputs must lie inside the env's action_space box."""
        env = RocketLandingEnv(cfg)
        mpc = MPCController(cfg)
        obs, _ = env.reset(seed=42)
        mpc.reset()
        action = mpc.predict(obs)
        assert env.action_space.contains(action)

    def test_gimbal_commands_are_exactly_zero(self, cfg) -> None:
        """Vertical channel only — the MPC never commands gimbal, matching the
        PID baseline's active scope so the T0 comparison stays fair."""
        mpc = MPCController(cfg)
        for altitude, descent in [(60.0, 20.0), (10.0, 5.0), (1.0, 2.0)]:
            action = mpc.predict(_make_obs(cfg, altitude, descent))
            assert action[1] == 0.0
            assert action[2] == 0.0

    def test_reset_clears_plan_and_warm_start(self, cfg) -> None:
        """After reset(), no plan is carried into the next episode."""
        mpc = MPCController(cfg)
        for _ in range(10):
            mpc.predict(_make_obs(cfg, 40.0, 12.0))
        assert mpc._plan is not None  # noqa: SLF001 (internal state check)
        assert mpc.solve_stats()["n_solves"] > 0

        mpc.reset()
        assert mpc._plan is None  # noqa: SLF001
        assert mpc._ticks_since_solve == 0  # noqa: SLF001
        assert mpc._last_throttle == 0.0  # noqa: SLF001
        assert mpc.solve_stats()["n_solves"] == 0

    def test_runs_full_episode_without_exception(self, cfg) -> None:
        """End-to-end smoke: env.reset → MPC.predict → env.step to termination."""
        env = RocketLandingEnv(cfg)
        mpc = MPCController(cfg)
        obs, _ = env.reset(seed=42)
        mpc.reset()

        for _ in range(int(cfg.env.episode.max_steps)):
            obs, _, terminated, truncated, _ = env.step(mpc.predict(obs))
            if terminated or truncated:
                break


class TestConfigValidation:
    @pytest.mark.parametrize(
        "override",
        [
            "mpc_controller.horizon=0",
            "mpc_controller.step_ticks=0",
            "mpc_controller.resolve_every_n_ticks=0",
        ],
    )
    def test_non_positive_cadence_rejected(self, override: str) -> None:
        with pytest.raises(ValueError, match="must all be >= 1"):
            MPCController(_compose([override]))

    def test_replan_interval_longer_than_plan_rejected(self) -> None:
        """A replan interval past the plan's own span would apply a move the
        optimiser never planned; that is a config error, not something to
        silently clamp."""
        cfg = _compose(
            [
                "mpc_controller.horizon=4",
                "mpc_controller.step_ticks=2",
                "mpc_controller.resolve_every_n_ticks=9",  # > 4 * 2
            ]
        )
        with pytest.raises(ValueError, match="exceeds the plan's own span"):
            MPCController(cfg)


class TestPredictionMatrices:
    def test_matrices_are_causal_and_correctly_shaped(self, cfg) -> None:
        """Step k may only depend on throttles applied at or before k."""
        mpc = MPCController(cfg)
        n = int(cfg.mpc_controller.horizon)
        S_z, f_z, S_v, f_v = mpc.build_prediction_matrices(50.0, 10.0, _total_mass(cfg), 1.0)

        assert S_z.shape == (n, n) and S_v.shape == (n, n)
        assert f_z.shape == (n,) and f_v.shape == (n,)
        assert np.allclose(S_z, np.tril(S_z))
        assert np.allclose(S_v, np.tril(S_v))
        # More throttle must mean less descent and less altitude lost.
        assert np.all(np.diag(S_v) < 0.0)
        assert np.all(np.diag(S_z) < 0.0)

    def test_matches_independent_integration_of_the_same_model(self, cfg) -> None:
        """The condensed profiles must equal a fine-step integration of the
        linearized continuous model under the same zero-order-hold input.

        This checks the exact-discretization constants *and* the condensation
        recursion against a method that shares none of their algebra.
        """
        mpc = MPCController(cfg)
        n = int(cfg.mpc_controller.horizon)
        delta = int(cfg.mpc_controller.step_ticks) / float(cfg.env.episode.control_hz)

        altitude, descent = 50.0, 10.0
        mass = _total_mass(cfg)
        d = cfg.env.dynamics
        drag_c = 0.5 * RHO_AIR_SEA_LEVEL * float(d.drag_coefficient) * float(d.reference_area_m2)
        damping = 2.0 * drag_c * abs(descent) / mass
        thrust_per_throttle = float(d.max_thrust_N) / mass
        gravity_eff = G_EARTH + drag_c * abs(descent) * descent / mass

        rng = np.random.default_rng(0)
        u = rng.uniform(0.0, 1.0, size=n)

        # Reference: explicit RK4 on ż = vz, v̇z = −a·vz − β·u + g_eff.
        substeps = 200
        h = delta / substeps
        z, vz = -altitude, descent
        z_ref, v_ref = np.zeros(n), np.zeros(n)
        for k in range(n):
            accel = gravity_eff - thrust_per_throttle * u[k]

            def deriv(state, accel=accel):
                return np.array([state[1], -damping * state[1] + accel])

            state = np.array([z, vz])
            for _ in range(substeps):
                k1 = deriv(state)
                k2 = deriv(state + 0.5 * h * k1)
                k3 = deriv(state + 0.5 * h * k2)
                k4 = deriv(state + h * k3)
                state = state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            z, vz = float(state[0]), float(state[1])
            z_ref[k], v_ref[k] = z, vz

        S_z, f_z, S_v, f_v = mpc.build_prediction_matrices(altitude, descent, mass, 1.0)
        assert np.allclose(S_z @ u + f_z, z_ref, atol=1e-6)
        assert np.allclose(S_v @ u + f_v, v_ref, atol=1e-6)

    def test_zero_drag_reduces_to_closed_form_free_fall(self, cfg) -> None:
        """At rest the damping term vanishes; the a→0 branch must reproduce the
        textbook constant-acceleration solution rather than divide by zero."""
        mpc = MPCController(cfg)
        n = int(cfg.mpc_controller.horizon)
        delta = int(cfg.mpc_controller.step_ticks) / float(cfg.env.episode.control_hz)
        altitude = 100.0

        S_z, f_z, S_v, f_v = mpc.build_prediction_matrices(altitude, 0.0, _total_mass(cfg), 1.0)
        u = np.zeros(n)  # engine off ⇒ pure free fall
        t = np.arange(1, n + 1) * delta

        assert np.allclose(S_v @ u + f_v, G_EARTH * t)
        assert np.allclose(S_z @ u + f_z, -altitude + 0.5 * G_EARTH * t**2)


class TestDescentEnvelope:
    def test_terminal_value_is_the_touchdown_target(self, cfg) -> None:
        """At the pad the envelope is exactly the touchdown speed target."""
        mpc = MPCController(cfg)
        at_pad = mpc.descent_envelope(0.0, _total_mass(cfg), 1.0)
        assert at_pad == pytest.approx(float(cfg.mpc_controller.envelope.touchdown_target_mps))

    def test_monotone_in_altitude_and_within_clip_bounds(self, cfg) -> None:
        mpc = MPCController(cfg)
        env_cfg = cfg.mpc_controller.envelope
        altitudes = np.linspace(0.0, 100.0, 60)
        envelope = mpc.descent_envelope(altitudes, _total_mass(cfg), 1.0)

        assert np.all(np.diff(envelope) >= 0.0)
        assert np.all(envelope >= float(env_cfg.min_descent_mps))
        assert np.all(envelope <= float(env_cfg.max_descent_mps))

    def test_negative_altitude_clamped(self, cfg) -> None:
        """Below the pad the square root must stay real."""
        mpc = MPCController(cfg)
        assert mpc.descent_envelope(-5.0, _total_mass(cfg), 1.0) == pytest.approx(
            mpc.descent_envelope(0.0, _total_mass(cfg), 1.0)
        )

    def test_heavier_vehicle_gets_a_more_conservative_envelope(self, cfg) -> None:
        """The substantive difference from a fixed tuned flare: less braking
        authority ⇒ a lower allowed descent rate, with no gain re-tuning."""
        mpc = MPCController(cfg)
        altitude = 5.0  # inside the clip bounds for both masses
        light = mpc.descent_envelope(altitude, _total_mass(cfg), 1.0)
        heavy = mpc.descent_envelope(altitude, 2.0 * _total_mass(cfg), 1.0)
        assert heavy < light

    def test_tilted_vehicle_gets_a_more_conservative_envelope(self, cfg) -> None:
        """Tilt costs vertical thrust authority, so the envelope tightens."""
        mpc = MPCController(cfg)
        altitude = 5.0
        upright = mpc.descent_envelope(altitude, _total_mass(cfg), 1.0)
        tilted = mpc.descent_envelope(altitude, _total_mass(cfg), 0.6)
        assert tilted < upright


class TestPlantInputProjection:
    def test_below_ignition_threshold_shuts_the_engine_off(self, cfg) -> None:
        mpc = MPCController(cfg)
        threshold = float(cfg.mpc_controller.ignition_threshold)
        assert mpc._project_to_plant_input(0.0) == 0.0  # noqa: SLF001
        assert mpc._project_to_plant_input(threshold * 0.5) == 0.0  # noqa: SLF001

    def test_dead_zone_is_raised_to_the_throttle_floor(self, cfg) -> None:
        """The plant cannot deliver thrust between zero and throttle_min, so a
        command in that band is raised to the floor rather than sent as-is."""
        mpc = MPCController(cfg)
        throttle_min = float(cfg.env.dynamics.throttle_min)
        threshold = float(cfg.mpc_controller.ignition_threshold)
        assert threshold < throttle_min, "fixture assumes a non-empty dead zone"
        assert mpc._project_to_plant_input(threshold) == throttle_min  # noqa: SLF001

    def test_admissible_commands_pass_through_and_saturate(self, cfg) -> None:
        mpc = MPCController(cfg)
        assert mpc._project_to_plant_input(0.5) == pytest.approx(0.5)  # noqa: SLF001
        assert mpc._project_to_plant_input(2.0) == pytest.approx(  # noqa: SLF001
            float(cfg.env.dynamics.throttle_max)
        )


class TestPhysicalSanity:
    def test_solution_is_a_bounded_plan_of_the_configured_length(self, cfg) -> None:
        mpc = MPCController(cfg)
        plan = mpc._solve(40.0, 10.0, _total_mass(cfg), 1.0)  # noqa: SLF001
        assert plan.shape == (int(cfg.mpc_controller.horizon),)
        assert np.all(plan >= 0.0) and np.all(plan <= 1.0)

    def test_coasting_high_and_slow_shuts_the_engine_off(self, cfg) -> None:
        """Well inside the envelope with altitude to spare, burning fuel is
        strictly wasteful — the plan should let the vehicle accelerate."""
        mpc = MPCController(cfg)
        throttle = mpc.predict(_make_obs(cfg, altitude_m=60.0, descent_rate_mps=1.0))[0]
        assert throttle == 0.0

    def test_fast_near_the_pad_commands_hard_braking(self, cfg) -> None:
        """Far outside the envelope with almost no height left: full authority."""
        mpc = MPCController(cfg)
        throttle = mpc.predict(_make_obs(cfg, altitude_m=5.0, descent_rate_mps=20.0))[0]
        assert throttle > 0.9

    def test_tracking_the_envelope_commands_more_than_hover(self, cfg) -> None:
        """Sitting exactly on the envelope still requires net deceleration to
        follow it down, so the command must exceed the weight-cancelling
        throttle without saturating."""
        mpc = MPCController(cfg)
        altitude = 10.0
        on_envelope = float(mpc.descent_envelope(altitude, _total_mass(cfg), 1.0))
        throttle = mpc.predict(_make_obs(cfg, altitude, on_envelope))[0]
        assert _hover_throttle(cfg) < throttle < 1.0


class TestMatrixInvariants:
    def test_identical_observations_give_identical_actions(self, cfg) -> None:
        """No randomness anywhere: the reproducibility guarantee the whole
        fixed-seed matrix rests on.

        The controller is deliberately *stateful* across a rollout (the plan and
        its warm start carry forward), so the claim under test is that a whole
        observation sequence replays identically — both on a second instance and
        on the same instance after ``reset()``.
        """
        observations = [_make_obs(cfg, 60.0 - 2.0 * i, 5.0 + 0.3 * i) for i in range(25)]

        mpc = MPCController(cfg)
        first = np.array([mpc.predict(o) for o in observations])

        other = MPCController(cfg)
        assert np.array_equal(np.array([other.predict(o) for o in observations]), first)

        mpc.reset()
        assert np.array_equal(np.array([mpc.predict(o) for o in observations]), first)

    def test_replans_on_the_configured_cadence(self, cfg) -> None:
        """The plan is reused between solves; solve count is ceil(ticks / k)."""
        interval = int(cfg.mpc_controller.resolve_every_n_ticks)
        n_ticks = 3 * interval + 2
        mpc = MPCController(cfg)
        for _ in range(n_ticks):
            mpc.predict(_make_obs(cfg, 40.0, 10.0))

        expected = -(-n_ticks // interval)  # ceil division
        assert mpc.solve_stats()["n_solves"] == expected

    def test_solve_stats_reported_before_any_solve(self, cfg) -> None:
        assert MPCController(cfg).solve_stats() == {
            "n_solves": 0,
            "solve_ms_mean": 0.0,
            "solve_ms_p99": 0.0,
        }

    def test_solve_stats_are_populated_after_solving(self, cfg) -> None:
        mpc = MPCController(cfg)
        for _ in range(10):
            mpc.predict(_make_obs(cfg, 40.0, 10.0))
        stats = mpc.solve_stats()
        assert stats["n_solves"] >= 1
        assert stats["solve_ms_mean"] > 0.0
        assert stats["solve_ms_p99"] >= stats["solve_ms_mean"]
