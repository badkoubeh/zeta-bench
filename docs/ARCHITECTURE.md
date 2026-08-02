# ZetaBench Architecture

## Overview

ZetaBench is a physics-first robustness characterization environment for control
policies. Any controller — deep RL, MPC, LQR, PID, or world-model-based — faces
an identical graduated disturbance matrix and produces reproducible, comparable
evidence of its failure modes.

The **rocket-landing environment** is the reference implementation. It is chosen
for well-defined physics and a clear success criterion, not as a claim about
deployed practice. (Real operators land via convex optimization, not RL.) The
same scaffolding is intended to host eVTOL/UAV and bipedal locomotion environments.

All controllers implement the same evaluation interface so they are scored under
identical conditions with identical seeds. This fair-comparison discipline is what
makes the robustness verdict credible.

---

## 1. Environment Fidelity

**Current:** Moderate fidelity — custom 6-DOF rigid body dynamics.

Parameters (mass, thrust, moment of inertia, drag coefficient) are grounded in publicly
documented Falcon-9 / Merlin-1D figures, sourced per parameter in
`docs/parameter_sources.md`. (Not RocketPy — its default vehicle is a small hobbyist
rocket; earlier revisions of this doc and the README claimed otherwise.) The dynamics
are derived from first principles in `notebooks/physics_derivation.ipynb` and
implemented in `dynamics/`. This is not a tutorial copy — the derivation is a core rigor
differentiator, and the notebook re-checks each derived term against the running code.

**Upgrade path:** High fidelity (full aerodynamics, gimbal actuator dynamics, fuel
mass depletion) is designed in from day one via an abstract base class. Upgrading
is a config change and a new dynamics subclass — no RL, environment, or evaluation
code changes required.

---

## 2. Dynamics Module — Extensibility Design

```python
class RocketDynamics(ABC):
    @abstractmethod
    def step(self, state: State, action: Action, dt: float,
             wind_velocity_ned: np.ndarray | None = None) -> State:
        ...

    @abstractmethod
    def get_params(self) -> DynamicsParams:
        ...
```

| Class | Status | Description |
|---|---|---|
| `ModerateFidelityDynamics` | Built Phase 1 | 6-DOF rigid body, Falcon-9-class params |
| `HighFidelityDynamics` | Future upgrade | Aero, gimbal, fuel depletion |

Shared value types (`State`, `Action`, `DynamicsParams`, quaternion helpers) live
in `dynamics/types.py` and are the only thing other packages read from `dynamics/`.

**Design rule:** Nothing outside `dynamics/` imports from it directly except `envs/`
(and read-only type imports elsewhere). Config flag `dynamics.fidelity: moderate |
high` controls which class is loaded at runtime.

---

## 3. Observation Space (17-dimensional)

| Group | Variables | Dim |
|---|---|---|
| Position (NED) | x, y, z | 3 |
| Velocity (NED) | vx, vy, vz | 3 |
| Attitude (Euler) | roll, pitch, yaw | 3 |
| Angular rates (body) | p, q, r | 3 |
| Last applied action | throttle, gimbal_pitch, gimbal_yaw | 3 |
| Fuel | fuel_mass, fuel_remaining | 2 |
| **Total** | | **17** |

The observation carries the **last applied action** (post actuator-delay), not a
thrust vector — it is the true actuator state the vehicle is responding to.

**Normalisation.** Observations are scaled by `FixedObsScaler`
(`utils/normalisation.py`) using fixed physical bounds from `env.obs_scaler`
(config-driven, deterministic — no running statistics, so no hidden state leaks
into the evaluation path). Internal physics stays `float64`; the observation is
emitted as `float32` (SB3/Gymnasium convention; MPS has no `float64`).

**Sensor noise** (Gaussian σ + sparse spikes) is added to the *scaled* observation
inside `RocketLandingEnv._build_obs`, driven by `set_disturbance` — so σ reads as a
fraction of each sensor's full-scale range. It is off by default and switched on
per-cell by the graduated matrix; nominal training observations are clean unless
domain randomization (§10) is enabled.

---

## 4. Action Space

**Continuous 3-vector** `[throttle, gimbal_pitch_command, gimbal_yaw_command]`:

| Index | Component | Range | Physical meaning |
|---|---|---|---|
| 0 | throttle | `[0, 1]` | scaled to `[throttle_min, throttle_max] × max_thrust_N` |
| 1 | gimbal_pitch_command | `[-1, 1]` | scaled to `±gimbal_max_rad` in dynamics |
| 2 | gimbal_yaw_command | `[-1, 1]` | scaled to `±gimbal_max_rad` in dynamics |

The space is deliberately **asymmetric** (`low = [0, -1, -1]`): a rocket engine
cannot produce negative thrust, and encoding that as a bound rather than a penalty
keeps the physics honest.

Rationale: throttle + 2-axis gimbal matches real thrust-vector-control actuation
and is the minimal authority set for 6-DOF landing. Naturally suits SAC (off-policy,
continuous action space).

**Actuator delay** is applied between command and physics: the vehicle responds to a
lagged command, and dynamics, reward, and the observation's `last_action` all use the
*applied* action.

---

## 5. Reward Function

**Hybrid: potential-based dense shaping + impact-aware sparse terminal**

```
R(t) = R_dense(t) + R_terminal (at episode end)

R_dense(t) — potential-based (PBRS: gamma·Phi(s') − Phi(s)), so shaping does not
             change the optimum. Phi(s) = −landing_cost(s), built from:
  - Distance to target pad (lateral + altitude, weighted)
  - Velocity magnitude penalty (encourage deceleration)
  - Near-pad landing-speed penalty, gated by exp(-altitude / ground_gate_altitude_m)
    (≈1 at touchdown, 0 aloft) — bleeds off speed during the flare; zero at the
    landed state, so the potential optimum is unchanged
  - Attitude / angular-rate deviation penalty (penalise tilt and spin)
  plus small non-potential regularisers: fuel burned and control effort

R_terminal — impact-aware, with enforced outcome ordering
(safe landing > slow upright crash > fast tilted crash > out-of-bounds):
  - success:       +success_bonus
  - crash:         −(crash_base_penalty + weighted touchdown speed² + tilt² +
                   angular-rate² + lateral²), clipped at terminal_clip_abs
  - timeout:       −(timeout_base_penalty + timeout_state_weight · landing_cost(s_T))
                   (truncation, not termination — the episode simply ran out of steps)
  - out_of_bounds: −out_of_bounds_penalty  (pinned as the worst outcome, above the
                   crash clip so fleeing the box can never be cheaper than crashing)
```

On terminal transitions the next-state potential is taken as 0 (absorbing-state
convention), so an episode's shaping telescopes to `−Phi(s_0)` and the outcome is
carried entirely by the terminal term — this is what removes the "crash early to
stop accumulating penalties" incentive.

The crash penalty is **not flat** — it scales with how hard/tilted the impact is;
the touchdown-speed term dominates, with the near-pad landing-speed shaping added,
to drive softer touchdowns that clear (not merely reach) the 3.0 m/s success gate
(`env.touchdown.velocity_threshold_mps`; success also requires ≤5° tilt and
≤~10°/s angular rate). All weights are config-driven (`configs/reward.yaml`,
companion `docs/reward_engineering.md`) — no hardcoded reward coefficients.

PBRS policy-invariance requires the shaping discount to match the agent's — the env
reads `cfg.agent.gamma` (falling back to 0.99 when built without an agent section).

---

## 6. Controllers

All controllers share the same evaluation interface — `predict(obs, deterministic)`,
`save`, `load` — so the robustness harness drives them identically regardless of the
underlying algorithm. Identical conditions, identical seeds. This fair-comparison
discipline is the mechanism that makes cross-paradigm results credible.

| Controller | Type | Status |
|---|---|---|
| PID baseline | Classical (cascaded) | ✅ Implemented (`configs/pid_controller.yaml`) |
| SAC agent | Off-policy RL | ✅ Trained (curriculum, γ=0.999) |
| PPO agent | On-policy RL | ✅ Trained (curriculum, γ=0.999) |
| LQR baseline | Classical (optimal) | Planned |
| MPC baseline | Optimization-based | Planned |

| Property | SAC | PPO |
|---|---|---|
| Type | Off-policy | On-policy |
| Sample efficiency | High | Moderate |
| Continuous actions | Natural fit | Requires tuning |
| Stability | Moderate | High |

**Controller variants.** The same architecture trained under different regimes
(nominal vs. robust — see §11) is a *variant*, not a new controller. Variants are
named in the matrix rows (`sac`, `sac_robust`) and share one robustness card (§8),
so the effect of robustness training is read with everything else held constant.

Tuned hyperparameter sets exported from an HPO sweep land in
`configs/agent/{agent}_tuned.yaml` (§12) and are selected with `agent=sac_tuned`.

---

## 7. Primary Robustness Evaluation — Graduated Disturbance Matrix

The primary evaluation mode. Every controller faces an identical, fixed-seed
disturbance matrix — conditions are held constant across controllers so results
are directly comparable and reproducible. This produces the signature heatmap
(disturbance type × severity × success rate).

Owned by `robustness/evaluation.py` (rollout + aggregation + grid orchestration);
`experiments/evaluate_robustness.py` is a thin Hydra entrypoint that builds the
controllers and wires outputs.

**The four disturbance types** (`robustness/disturbances.py`) — a `Disturbance` is an
immutable value object whose all-zero default *is* the nominal condition, so types are
independently testable AND composable:

| Disturbance | Where it enters | Swept levels (`configs/eval.yaml`) |
|---|---|---|
| Wind | Physics, as relative airspeed `v_air = v_rocket − v_wind` in the drag term | Magnitude [0, 2, 5, 10] m/s × direction [0, 90, 180, 270, 45]° |
| Mass uncertainty | Env rebuilds dynamics params from the immutable nominal set | Dry-mass offset [−20%, −10%, 0, +10%, +20%] |
| Sensor noise | Added to the scaled observation (Gaussian + sparse spikes) | σ [0, 0.01, 0.05, 0.1] × spike probability [0, 0.01, 0.05] |
| Actuator delay | Integer control-tick latency on the applied action | Implemented and unit-tested; **not yet a swept axis** |
| Combined | All of the above simultaneously | Max level, single cell |

**Fairness invariants** (enforced in `robustness/evaluation.py`):
- Per (controller, cell), the per-episode seed stream is re-derived from the same
  master `cfg.seed` — every controller sees identical initial conditions *and*
  identical sensor-noise realisations within a cell.
- Initial conditions are pinned (`env.curriculum.schedule: fixed`,
  `task_difficulty: 1.0`) so `disturbance_severity` is the sole independent
  variable. A non-fixed schedule is flagged at runtime.
- Sampling: `eval.seeds` × `eval.episodes_per_seed` (default 5 × 20 = 100 episodes
  per cell).

**Outputs.** A long-format CSV (`results/robustness_matrix.csv`) is the source of
truth — one row per (controller, cell) with success rate, outcome counts
(success / crash / out-of-bounds / timeout), return mean+std, mean touchdown speed,
mean fuel used, and mean episode length. `robustness/heatmap.py` renders the hero
figure: one panel per controller, disturbance type × *ordinal* severity level
(severity scales differ per type, so each cell is annotated with its actual value).
Wind averages over direction and sensor noise over spike probability; the full
per-direction / per-spike detail stays in the CSV. Per-cell sample counts are
printed in the caption — a small n must not read as precision.

---

## 8. Robustness Cards — the within-controller verdict

Where the heatmap answers *"how do these controllers compare?"*, a **card**
(`robustness/cards.py`, entrypoint `experiments/robustness_card.py`, config
`configs/robustness_card.yaml`) answers the mission question for one controller:
*"is it robust enough to deploy, and at what disturbance magnitude does it break?"*

- One panel per disturbance family: a **degradation curve** (success rate vs. severity).
- A **break-point**: the smallest tested severity whose mean success rate falls below
  the deployment gate (`robustness_card.gate`, default 0.95).
- One line per **variant** (e.g. `nominal-trained` vs. `robust`), so robustness
  training is read directly — same architecture, same cells, same seeds, training
  distribution the only delta. Variants missing from the loaded rows are skipped, so
  cards render before every variant exists.

Cards consume the same long-format matrix rows as the heatmap and follow its
aggregation conventions (wind over direction, sensor noise over spike probability;
mass offset keeps its sign, and its break-point is the smallest *magnitude* that
fails on either side). Variant runs write their own CSV so the primary matrix is
never overwritten. Cards **complement, never replace** the heatmap — a break-point
is only interpretable against a reference controller, and the graduated fixed-seed
matrix remains the fairness anchor. Artifacts land in `results/cards/{card}.{png,json}`.

---

## 9. Optional — Adversarial / Worst-Case Disturbance Search

A learned adversary (SB3 SAC) searches for the disturbance within physical bounds
that most reliably breaks a given controller. This is a stress-test for a *single*
controller, not a cross-controller comparison tool. An adaptive adversary fights
each controller differently, so its findings are not comparable across controllers.
Adversarial results are always reported separately from the graduated matrix.

**Adversary action space:**
```
a_adv = [wind_x, wind_y, wind_z,    # wind force vector
         noise_magnitude,             # sensor noise scale
         mass_offset]                 # payload mass variance
```

**Training loop:**
```
for each episode:
    agent acts → adversary observes state → adversary injects disturbance
    agent reward: landing success
    adversary reward: -agent_reward (zero-sum)
    alternate gradient updates: N agent steps, 1 adversary step
```

**Status.** Scaffolded but **not wired**: `adversary/adversary_policy.py` raises
`NotImplementedError` and `experiments/train.py` refuses `train_mode=adversarial`.
Domain randomization (§10) is the supported way to harden a policy today, and remains
the fallback if adversarial training later destabilises learning.

---

## 10. Curriculum and Domain Randomization

Two independent training-time axes. Keeping them decoupled is what keeps the
training curriculum and the graduated disturbance matrix independent (see the
"Naming conventions" section of `CONTRIBUTING.md`).

**Task difficulty — the nominal problem's hardness** (`envs/curriculum.py`):

```
task_difficulty = f(global_step)  # 0.0 → 1.0 (linear over anneal_steps, clamped)

altitude         ~ U(min_height,  min_height + d·(max_height − min_height))
lateral_offset   ~ U(−d·max_lateral, +d·max_lateral)
descent_velocity ~ U(min_descent, min_descent + d·(max_descent − min_descent))
attitude         = upright   # attitude_tilt_max_rad = 0 today; reserved knob
angular_rate     = 0
```

`task_difficulty` scales the **initial-condition envelope only**. It deliberately
does NOT scale environmental disturbances (`disturbance_severity`) or adversary
weight. Scheduler is config-driven (`schedule: linear | fixed`; evaluation pins
`fixed` because its `_global_step` is zero); the current value is logged to wandb as
`curriculum/task_difficulty`.

**Disturbance severity — domain randomization** (`envs/domain_randomization.py`), a
config-gated training-env wrapper (`env.domain_randomization`, **off by default**):
each training episode samples a fixed disturbance from configured ranges via
`set_disturbance`. `severity_anneal_steps` scales the sampled magnitudes 0→1 over N
**per-env** steps, mirroring `curriculum.anneal_steps`. Run together, the two anneals
phase in harder initial conditions and harder disturbances on parallel schedules, so
a cold-start policy masters the nominal task before facing wide randomisation.

Ranges live in `configs/env.yaml` (single source of disturbance magnitudes) and span
the *learnable* portion of the robustness grid — the σ ≥ 0.10 noise regime is a shared
physics wall and is intentionally excluded. Unlike the adversary, DR is not adaptive,
and it wraps the **training env only** — it never touches the graduated matrix, whose
per-cell severity is fixed and seeded rather than sampled.

---

## 11. Progressive Training Profile — staged training with a verification gate

A one-shot pipeline that produces the *robust* controller variants (§6, §8).
Sequencing lives in `robustness/progressive.py`, the gate in
`robustness/verification.py`, the entrypoint in `experiments/train_profile.py`,
budgets in the `profile` config group (`configs/profile/{progressive,smoke}.yaml`).

```
per agent (SAC, then PPO):
  Stage A (naive)  → verification gate → Stage B (robust)
```

| Stage | What runs | Key knobs (progressive profile) |
|---|---|---|
| **A — naive** | Task-difficulty curriculum ramps 0→1, DR **off** | 2M global steps, ramp over 80% of the per-env budget |
| **Gate** | Candidates archived, then evaluated under nominal conditions | threshold 0.90 success over 100 episodes at `task_difficulty` 0.4 |
| **B — robust** | Resume gate-selected checkpoint, curriculum pinned, DR severity ramps 0→1 | 2M global steps, ramp over 80% of the per-env budget |

Design points that are easy to get wrong, and are therefore load-bearing:

- **The gate difficulty is 0.4, not 1.0.** A well-trained agent is ~99% at 0.4 but
  degrades sharply beyond (~52% at 1.0), so a 90% bar at full difficulty is
  effectively unreachable. Stage A's model-selection eval (`eval_task_difficulty`)
  is kept in sync with the gate so selection targets what the gate measures.
- **Candidates are archived before scoring.** An extension run re-creates SB3's
  `EvalCallback` with `best_mean_reward = −inf`, whose first evaluation would
  otherwise overwrite `best_model.zip` with something worse than what the gate
  scored. The archive carries `replay_buffer.pkl` when present so a SAC resume warms
  its buffer (the buffer belongs to the stage's *final* model — an accepted
  approximation for an off-policy learner).
- **A failed gate is bounded, not fatal.** Up to `max_extensions` (2) Stage A
  extension runs of 500k steps each; then that agent's chain is aborted, recorded in
  `gate_report.json`, and the next agent still runs.
- **Step-count units.** `total_steps` is *global* across `compute.n_envs` workers;
  the anneal knobs are *per-env*. `per_env_anneal_steps` does the conversion. Per-env
  counters reset whenever envs are rebuilt, which is why extension and Stage B runs
  pin `schedule=fixed` rather than re-ramping.
- **Stage B's eval env stays nominal** (`eval_task_difficulty: 1.0`, no DR) so
  model selection is comparable across stages.

Each stage is an ordinary `experiments/train.py` subprocess — its own wandb run,
checkpoints, and eval — so every stage remains individually reproducible.
`configs/profile/smoke.yaml` is the minutes-scale wiring check.

---

## 12. Hyperparameter Optimisation

Optuna via the Hydra sweeper (`configs/hpo_sac.yaml`, `configs/hpo_ppo.yaml`;
TPE sampler seeded from `cfg.seed`, objective = best eval mean reward). Each trial is
one `experiments/train.py` run, made unique by a `_t{job.num}` run-name suffix. Sweep
budget (total steps, trial count, eval cadence) comes from the `budget` group
(`configs/budget/{full,laptop}.yaml`).

`experiments/export_best_params.py` bakes the winning trial into
`configs/agent/{agent}_tuned.yaml` for reuse on a single run. Two constraints shape it:

- **Algorithm vs. hardware split.** Only hardware-independent `agent.*` params
  (learning_rate, gamma, tau, ent_coef, …) are baked. Hardware-coupled `compute.*`
  params (batch_size, n_steps) are *reported*, not frozen — a batch size tuned on a
  GPU host must not be silently reused on a laptop.
- **No study DB.** Optuna 2.10 is incompatible with SQLAlchemy 2.x, so sweeps use
  in-memory storage and the per-sweep `optimization_results.yaml` is the persisted
  source of truth.

---

## 13. Module Boundaries

```
zeta-bench/
├── configs/                   # All hyperparams — nothing hardcoded elsewhere
│   ├── train.yaml             # + train_profile.yaml, hpo_{sac,ppo}.yaml
│   ├── env.yaml               # dynamics, episode, touchdown, curriculum, DR, obs scaler
│   ├── reward.yaml
│   ├── eval.yaml              # the graduated disturbance grid
│   ├── eval_{pid,rl,robustness,robustness_profile}.yaml
│   ├── robustness_card.yaml
│   ├── pid_controller.yaml
│   ├── adversary.yaml
│   ├── agent/                 # sac, ppo, pid, sac_tuned
│   ├── compute/               # cpu, mps, small_gpu, large_gpu, multi_gpu, kaggle_gpu
│   ├── budget/                # full, laptop  (HPO sweep budgets)
│   └── profile/               # progressive, smoke  (staged-training budgets)
├── dynamics/                  # Self-contained — only envs/ imports the classes
│   ├── base.py                # RocketDynamics ABC
│   ├── moderate_fidelity.py   # ModerateFidelityDynamics
│   ├── equations_of_motion.py
│   └── types.py               # State/Action/params value types (read-only elsewhere)
├── envs/                      # Gymnasium wrapper, consumes dynamics/
│   ├── rocket_landing_env.py
│   ├── curriculum.py          # task-difficulty annealing + IC sampler
│   ├── domain_randomization.py# training-only disturbance-sampling wrapper
│   └── reward.py              # potential, shaping, terminal
├── controllers/               # All controllers — same predict/save/load interface
│   ├── pid_baseline.py
│   ├── sac_agent.py
│   └── ppo_agent.py
├── robustness/                # The product: disturbances + evaluation + verdicts
│   ├── disturbances.py        # typed, composable disturbance value objects + grid
│   ├── evaluation.py          # graduated matrix runner (PRIMARY path)
│   ├── heatmap.py             # signature type × severity × success-rate figure
│   ├── cards.py               # per-controller degradation curves + break-points
│   ├── progressive.py         # Stage A → gate → Stage B sequencing
│   └── verification.py        # checkpoint archiving + gate selection
├── adversary/                 # Adversary policy + disturbance action space (scaffold)
│   └── adversary_policy.py
├── utils/                     # Leaf helpers — importable anywhere; never imports upward
│   ├── normalisation.py       # FixedObsScaler
│   ├── reproducibility.py     # seeded RNG factory
│   ├── logging_config.py, wandb_setup.py, sb3_callbacks.py, render.py
├── experiments/               # Entrypoints only — orchestration, no logic
│   ├── train.py, train_profile.py
│   ├── evaluate_pid.py, evaluate_rl.py, evaluate_robustness.py
│   ├── robustness_card.py, export_best_params.py
│   └── sagemaker_launch.py
├── scripts/check_diagram_sync.py  # pre-commit guard on the README control diagram
├── notebooks/                 # physics_derivation.ipynb, kaggle_train.ipynb
├── tests/                     # Physics correctness, reward sanity, obs bounds, wiring
└── results/                   # Artifacts: matrix CSV, heatmap, cards, videos
```

**Dependency rule:**
`configs ← dynamics ← envs ← controllers/adversary/robustness ← experiments`,
with `utils/` as a leaf importable by any of them.

Two rules carry the weight:
- **`robustness/` must never import from `experiments/`.** Entrypoints are
  orchestration; the evaluation logic has to be importable and testable without them.
  (`robustness/progressive.py` and `verification.py` invert this via *injected*
  runners — `StageRunner` / `GateEvaluator` — supplied by `train_profile.py`.)
- **`utils/` sits at the bottom of the graph** — it must not import from `envs/`,
  `controllers/`, `adversary/`, or `experiments/`.

No upward imports.

---

## 14. Execution & Reproducibility Infrastructure

| Layer | Mechanism |
|---|---|
| Local / CI runs | `Dockerfile` + `docker-compose.yml` (`train`, `eval`, `shell` services) |
| Convenience targets | `Makefile` — `make train`, `train-profile`, `eval`, `eval-pid`, `viz`, `test`, `lint`, `lock`, `install-hooks` |
| Hardware selection | `compute/` config group — `n_envs`, batch sizes, device, per host class |
| Cluster fan-out | `experiments/sagemaker_launch.py` (SAC is single-process off-policy: multi-GPU does not speed up *one* run — the fan-out is across seeds/configs) |
| Dependency pinning | `requirements.lock` (Linux x86_64 / Python 3.12), regenerated via `make lock` |
| Seed control | `utils/reproducibility.make_rng` everywhere — never bare `np.random.seed` |
| Experiment tracking | wandb; mode resolved from the environment (`utils/wandb_setup.py`), offline-safe |

---

## 15. Production Standards

| Standard | Implementation |
|---|---|
| Type hints | All functions, all modules |
| Docstrings | NumPy-style, including units |
| Config management | Hydra — single YAML tree, no hardcoded values |
| Experiment tracking | wandb — every run logged, Optuna sweeps for hyperparams |
| Reproducibility | Pinned lockfile, seed control, single-command Hydra entrypoints |
| Testing | pytest — physics correctness, edge cases, properties, reward, disturbances, wiring |
| CI | `coverage.yml` (pytest + `--cov-fail-under=90` on a bare 3.12 runner, ~30 s), `docker.yml` (image build + in-image pytest), `diagram-check.yml` |
| Pre-commit | `scripts/check_diagram_sync.py` blocks architecture-relevant commits that leave the README control diagram unstaged (`SKIP_DIAGRAM_CHECK=1` to bypass) |

---

## 16. Upgrade Paths

| Upgrade | What changes | What stays the same |
|---|---|---|
| High fidelity dynamics | `HighFidelityDynamics` class + obs space extension | Everything else |
| LQR / MPC baselines | New `controllers/` module implementing `predict` | Env, matrix, cards, heatmap |
| Actuator-delay sweep axis | Levels in `configs/eval.yaml` + a heatmap row | Disturbance model (already implemented) |
| eVTOL / UAV environment | New `UAVDynamics`, new env wrapper | All eval framework, controllers |
| CARLA simulator | Replace Gymnasium env | Agents, adversary, evaluation |
| Transformer policy | Swap MLP in SAC/PPO for transformer | Training loop, reward, curriculum |
