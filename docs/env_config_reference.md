# Environment Configuration Reference

Companion to `configs/env.yaml`. Explains every parameter group, the physics
behind each value, and how parameters interact. Read alongside
`docs/research_notes.md` (control-theory analysis) and `docs/ARCHITECTURE.md`
(system topology).

---

## Frame conventions

All coordinates and vectors in this codebase use one of two frames. Every
number in `env.yaml` is implicitly in one of them.

### NED — inertial frame

```
         N (+X)
         │
W ───────●──────── E (+Y)
         │
         D (+Z) ↓  gravity points here
```

**North / East / Down.** Z increases *downward*, so gravity is `+g` in Z.
A rocket hovering above the pad has a **negative** Z coordinate. Touchdown
is detected when `z ≥ 0` — the rocket has reached or passed pad level.

### FRD — body frame

**Forward / Right / Down.** Axes fixed to the rocket hull: +X points out
the nose, +Y points to the right, +Z points toward the belly. When the
rocket is upright (nose pointing skyward), body +X is aligned with
inertial −Z.

The attitude quaternion stored in the state rotates vectors from FRD to
NED. The "perfectly upright" quaternion is
`q = (√2/2, 0, √2/2, 0)` — a +90° rotation about the inertial Y axis
(see `dynamics/types.py::UPRIGHT_QUAT`).

---

## `env.dynamics` — vehicle model

### Masses

```yaml
dry_mass_kg: 25_000.0
initial_fuel_kg: 5_000.0
```

- **`dry_mass_kg`** — structural mass with the engine, plumbing, and avionics
  but no propellant. This never changes during a simulation run.
- **`initial_fuel_kg`** — propellant loaded for the landing burn only (not the
  full ascent load). Total mass at episode start = 30,000 kg.

Every force calculation uses `total_mass = dry_mass + fuel_mass(state)`.
As fuel burns, total mass decreases, so the same throttle command produces
more acceleration late in the burn. The moderate-fidelity model does **not**
update the inertia tensor as mass changes — that is an explicit simplification
documented in `README.md §Limitations`.

### Thrust and specific impulse

```yaml
max_thrust_N: 845_000.0
isp_s: 282.0
```

- **`max_thrust_N`** — the physical ceiling on engine output. Actual thrust at
  any step is `throttle × max_thrust_N`, where throttle ∈ [throttle_min,
  throttle_max].

- **`isp_s`** — specific impulse: the rocket-engine equivalent of fuel economy.
  One kilogram of propellant produces `isp_s` Newton-seconds of impulse. Higher
  = more efficient. 282 s is realistic for a kerosene/LOX engine at sea level
  (Merlin-1D reference value).

  It appears in the **Tsiolkovsky fuel-burn surrogate** used by the dynamics:

  ```
  ṁ_fuel = −Thrust / (Isp × g₀)        g₀ = 9.80665 m/s²
  ```

  At full throttle (845,000 N): burn rate ≈ 305 kg/s. The 5,000 kg reserve
  lasts roughly 16 seconds at full throttle — episodes are propellant-limited
  if the controller is aggressive.

**Thrust-to-weight ratio (TWR).** A critical derived quantity not directly in
the config but governing all control decisions:

```
TWR = Thrust / (total_mass × g)
```

| Throttle | Thrust (N) | TWR (full fuel, 30,000 kg) | Effect |
|---|---|---|---|
| 0 (engine off) | 0 | 0 | Free-fall, +9.81 m/s² downward |
| 0.4 (minimum) | 338,000 | 1.15 | Net deceleration, −2.3 m/s² |
| 1.0 (full) | 845,000 | 2.87 | Strong deceleration, −18.4 m/s² |

TWR > 1 means the rocket decelerates a descent (or accelerates upward).
TWR < 1 means it accelerates downward (falls faster).

**Hover throttle.** The throttle at which TWR = 1 exactly — zero net
vertical acceleration, velocity held constant:

```
u_hover = total_mass × g / max_thrust_N ≈ 30000 × 9.81 / 845000 ≈ 0.32
```

This value shifts slightly as fuel burns (0.29 at dry mass, 0.35 at full
fuel). Critically, `u_hover ≈ 0.32` falls **inside the deadband** (0, 0.4)
between engine-off and the minimum sustainable throttle. No steady throttle
can hold a constant velocity — the engine must duty-cycle (see
`throttle_min` below and `docs/research_notes.md §9.1`).

### Aerodynamics

```yaml
drag_coefficient: 0.75
reference_area_m2: 10.5
```

Drag force opposes the velocity vector, magnitude growing with speed squared:

```
F_drag = −½ × ρ × |v|² × Cd × A_ref     (direction: −v̂)
```

Sea-level air density ρ = 1.225 kg/m³ (constant throughout — altitude
variation is a known omission in moderate fidelity). At `v = 20 m/s`:

```
|F_drag| = 0.5 × 1.225 × 400 × 0.75 × 10.5 ≈ 1,929 N
```

Compared to gravity at full fuel (294,300 N), drag is under 1% at entry
speed and is therefore primarily a small damping term rather than a dominant
force at landing-burn velocities.

`reference_area_m2 = π × r²` with `r ≈ 1.83 m` — the cross-sectional area
of the vehicle nose-on. The actual drag depends on angle-of-attack and Mach
number, both ignored in moderate fidelity.

### Inertia tensor

```yaml
inertia_xx: 2.5e6    # kg·m²  pitch
inertia_yy: 2.5e6    # kg·m²  yaw
inertia_zz: 1.2e5    # kg·m²  roll
```

Rotational inertia resists angular acceleration exactly as mass resists
linear acceleration: `τ = I × α`. The tensor here is diagonal (principal
axes coincide with the body frame) — a valid approximation for a
rotationally-symmetric cylinder.

**Why Izz ≪ Ixx = Iyy.** The stage is a long, thin tube (~47 m tall,
~3.7 m diameter).

- **Pitch / yaw (Ixx, Iyy):** rotating end-over-end. Most of the mass
  (tanks, engine) is far from the rotation axis → high inertia. 2.5 × 10⁶
  kg·m² → at full gimbal torque ≈ 1.1 MN·m, angular acceleration ≈ 0.44 rad/s².
- **Roll (Izz):** spinning around the long axis. All mass is close to the
  axis → low inertia. Izz/Ixx ≈ 0.05 — roll is ~20× easier to spin.

**Control consequence.** The single gimballed engine generates zero roll
torque (the lever arm is along the body +X axis — `r × F` has no X
component). Roll is therefore structurally uncontrollable from the action
space. Any initial roll rate persists indefinitely in the moderate-fidelity
model. This is accepted as a scope limit; real vehicles use RCS thrusters
for roll. See `docs/research_notes.md §4`.

### Geometry

```yaml
engine_lever_arm_m: 15.0
```

The distance from the centre of mass to the gimbal pivot point. Torque =
force × moment arm. When the gimbal tilts the thrust vector by angle ε,
the lateral force component `F_lateral = T × sin(ε)` creates torque
`τ = L × F_lateral`. Appearing in
`dynamics/equations_of_motion.py::compute_gimbal_torque_body`:

```
τ_body = (0,  L × F_z_body,  −L × F_y_body)
```

Larger lever arm → more torque authority → faster attitude correction →
but also more sensitivity to gimbal noise.

### Actuator limits

```yaml
gimbal_max_rad: 0.0873    # ±5°
throttle_min: 0.4
throttle_max: 1.0
```

- **`gimbal_max_rad`** — the physical stop on the gimbal joint. Agent
  commands in `[−1, 1]` are scaled to `[−0.0873, +0.0873]` rad inside
  `compute_thrust_force_body`. This limits both the lateral force authority
  and therefore the attitude correction bandwidth.

- **`throttle_min`** and **`throttle_max`** — combustion stability limits.
  The engine cannot sustain ignition below 40% of rated thrust.
  `clamp_throttle` in `dynamics/equations_of_motion.py` implements the
  mapping:

  ```
  command ≤ 0          → thrust = 0  (engine off)
  command ∈ (0, 0.4)   → thrust = 0  (deadband — engine cannot run here)
  command ∈ [0.4, 1.0] → thrust = command × max_thrust_N
  ```

  The feasible throttle set is therefore `{0} ∪ [0.4, 1.0]`. Since
  `u_hover ≈ 0.32` lies in the gap, the controller must duty-cycle the
  engine to achieve an average thrust near hover — a limit cycle is
  physically unavoidable, not a tuning failure.

---

## `env.episode` — time and integration

```yaml
control_hz: 50
physics_substeps: 4
max_steps: 2500
```

Two time scales coexist:

| Loop | Frequency | dt | Role |
|---|---|---|---|
| Control | 50 Hz | 0.02 s | Agent / PID observes and outputs one action per tick |
| Physics | 200 Hz | 0.005 s | Dynamics integrate via RK4, action held constant (ZOH) |

The physics substep rate reduces numerical integration error without
increasing control latency. At 200 Hz the RK4 step is 5 ms, which gives
O(dt⁴) ≈ O(10⁻¹⁰) global error per step — adequate for the 50 s episode
length.

`max_steps = 2500` gives a 50 s wall-clock budget. An episode that has
not terminated (success / crash / out-of-bounds) by step 2500 is
**truncated** with outcome `timeout`. The budget is sized so a constant
2 m/s descent from `altitude_max = 60 m` (30 s) has 20 s of margin for
the deceleration phase.

---

## `env.init_conditions` — episode starting state

```yaml
altitude_min_m: 30.0
altitude_max_m: 60.0
lateral_offset_max_m: 50.0
descent_velocity_min_mps: 5.0
descent_velocity_max_mps: 20.0
attitude_tilt_max_rad: 0.0
```

At episode start the sampler draws from a curriculum-lerped envelope.
At difficulty `p ∈ [0, 1]` the active range for each axis is:

```
active_max = min_value + p × (max_value − min_value)
sample     = Uniform(min_value, active_max)
```

At `p = 0` (easiest) every axis is pinned to its minimum. At `p = 1`
(hardest) the full range is live.

**`altitude_min/max_m`** — height above the landing pad in metres.
Converted to NED Z in the curriculum sampler:

```python
position_NED = [x, y, -altitude]   # negative because Z-down, rocket is above pad
```

Current range [30, 60] m is sized for the altitude-only PID baseline (no
outer altitude loop). At `target_descent = 2 m/s`, 60 m takes 30 s — within
the 50 s budget. These values should be raised once the cascade outer loop
is wired in.

**`lateral_offset_max_m`** — maximum horizontal displacement from the pad
at episode start. Zero at `p = 0` (rocket starts directly above pad);
up to ±50 m at `p = 1`. The lateral and attitude loops are not yet active
in the PID baseline, so a nonzero lateral offset will result in a missed
landing until those loops are wired in.

**`descent_velocity_min/max_mps`** — initial downward velocity in NED Z
(positive = descending). The rocket always starts descending; the PID must
decelerate it from 5–20 m/s to the 3 m/s touchdown threshold.

At the hardest curriculum combination (20 m/s entry, 30 m altitude), the
rocket needs to decelerate 18 m/s. At max throttle deceleration ≈ 18.4 m/s²,
this takes about 1 s and uses roughly 9 m of altitude — leaving 21 m of
margin from `altitude_min = 30 m`.

**`attitude_tilt_max_rad`** — maximum tilt from vertical at episode start.
Currently 0 (always starts perfectly upright). Will be raised via curriculum
once the attitude loops are active.

---

## `env.oob` — out-of-bounds termination

```yaml
cylinder_radius_m: 200.0
ceiling_m: 600.0
```

The valid flight volume is a vertical cylinder centred on the pad:

```
lateral radius:  √(x² + y²) < 200 m
altitude:        0 < -z < 600 m   (i.e., z ∈ (−600, 0) in NED)
```

Leaving the cylinder on any face terminates the episode immediately with
outcome `out_of_bounds` (penalty: `reward.terminal.out_of_bounds_penalty`,
currently the worst terminal outcome — above any realistic crash — so a
policy can never make fleeing the box cheaper than a hard landing). This
prevents runaway trajectories from running the full 2500-step budget.

The ceiling check is `z < −600` in `rocket_landing_env.py::_check_termination`.
Note `position_z_m = 600` in the obs scaler matches this boundary — a rocket
at the ceiling maps to scaled z = −1.

---

## `env.touchdown` — success vs crash discrimination

```yaml
velocity_threshold_mps: 3.0
tilt_threshold_rad: 0.0873           # 5°
angular_rate_threshold_rad_s: 0.1745 # ~10°/s
```

When `z_NED ≥ 0` the rocket has reached pad level. All three conditions
are checked simultaneously:

| Condition | Threshold | Measured as |
|---|---|---|
| Speed | ≤ 3.0 m/s | `‖velocity_NED‖` — total 3D speed, not just vertical |
| Tilt | ≤ 5° | Angle between body +X (nose) and inertial −Z (up direction) |
| Spin | ≤ 10°/s | `‖angular_rate_body‖` |

All three satisfied → `success` (terminal `success_bonus`). Any one violated
→ `crash`, whose penalty is **impact-aware** — it scales with touchdown speed,
tilt, angular rate, and lateral error rather than a flat value, so a slow
upright crash is penalised less than a fast tilted one. The exact weights live
in `configs/reward.yaml::reward.terminal` (see `docs/reward_engineering.md`);
the touchdown-speed term is the dominant one and was recently strengthened.

The speed threshold was loosened from 2.0 to 3.0 m/s so a controller that
arrives nearly stopped is not failed on a fraction of a m/s; the impact-aware
crash penalty (and the near-pad landing-speed shaping in the reward) still
push the policy toward the softest touchdown it can achieve.

The speed threshold is intentionally the **3D norm**: lateral velocity at
touchdown counts. A rocket arriving at 1 m/s downward but 3 m/s sideways
exceeds the threshold and crashes.

Tilt is computed via a direct quaternion projection in
`envs/reward.py::tilt_from_vertical` — this avoids the gimbal-lock
singularity that Euler-angle decomposition has at the perfectly-upright pose
(pitch = 90°, which is the nominal landing attitude).

---

## `env.disturbance` — static single-run disturbance

```yaml
disturbance:
  wind_magnitude_mps: 0.0
  wind_direction_deg: 0.0        # 0=N, 90=E, 180=S, 270=W
  mass_offset_fraction: 0.0      # signed: +0.20 = 20% heavier than nominal
  sensor_noise_sigma: 0.0        # Gaussian σ on the scaled observation
  sensor_spike_probability: 0.0  # per-component, per-step spike probability
  sensor_spike_magnitude: 0.5    # scaled-obs units; spike size when one fires
  actuator_delay_steps: 0        # control-tick latency on the applied action
```

A **single, fixed** disturbance applied for the whole run. These values are
only the standalone / manual-eval defaults (all zero = nominal dynamics). The
graduated robustness matrix overrides this block per cell at runtime via
`RocketLandingEnv.set_disturbance(...)`, so a matrix sweep does not read these
numbers — they matter only when you launch the env directly without the
matrix. Wind is given in **polar** form (magnitude + compass bearing) and
converted to a NED velocity; see `robustness/disturbances.py`.

The four disturbance channels map to distinct physical failure modes:

| Channel | Physical meaning | Enters the model via |
|---|---|---|
| `wind_*` | Steady crosswind → relative-airspeed drag | drag term in `equations_of_motion` |
| `mass_offset_fraction` | Dry-mass estimate error (payload/residual-fuel uncertainty) | rebuilds dynamics with `dry_mass × (1 + fraction)` |
| `sensor_noise_sigma` / `sensor_spike_*` | Gaussian noise + rare spikes on the scaled observation | `_apply_sensor_noise` |
| `actuator_delay_steps` | Control-tick latency between command and actuation | delay `deque` buffer |

## `env.domain_randomization` — training-time randomisation wrapper

```yaml
domain_randomization:
  enabled: false
  severity_anneal_steps: 0             # per-env steps to ramp severity 0→1 (0 = no ramp)
  wind_magnitude_mps: [0.0, 10.0]      # sampled magnitude; direction is uniform 0–360°
  mass_offset_fraction: [-0.20, 0.20]  # signed dry-mass offset fraction
  sensor_noise_sigma: [0.0, 0.03]      # Gaussian σ on the scaled observation
  sensor_spike_probability: [0.0, 0.03]
  sensor_spike_magnitude: 0.5          # scaled-obs units; spike size when one fires
  actuator_delay_steps: [0, 0]         # integer control-tick latency range (inclusive)
```

Unlike `env.disturbance` (one fixed disturbance for the whole run), this is a
**training-only Gymnasium wrapper** (`envs/domain_randomization.py`,
`wrap_if_enabled`). When `enabled`, every *training* episode draws a fresh
disturbance uniformly from the ranges above at `reset()` and pushes it into the
env through the same `set_disturbance` hook. The policy therefore learns across
the disturbance *distribution* instead of only nominal dynamics — the standard
robustness remedy for a partially-observed disturbed environment.

Key properties, and why they matter for reproducibility and fair comparison:

- **Off by default** (`enabled: false`), so nominal training and *all*
  evaluation paths are unchanged unless a training config explicitly turns it on.
- **Training env only.** It is wired into the PPO/SAC training vec-envs, not the
  eval / model-selection env and not the graduated robustness matrix — those stay
  deterministic per cell so model selection and cross-controller comparison
  happen on identical, clean conditions.
- **Isolated, seeded RNG.** The wrapper samples from its own
  `np.random.default_rng` (`_dr_rng`), separate from the env's IC/sensor-noise
  stream, and reseeds it deterministically from `reset(seed=...)`. Disturbance
  sampling is reproducible and does not perturb the env's other randomness.
- **Ranges span the learnable portion of the robustness grid.** The extreme
  `σ ≥ 0.10` sensor-noise regime is intentionally excluded — it is a shared
  physics wall no controller survives, so training on it only adds noise.
- **`actuator_delay_steps: [0, 0]`** means delay is *not* randomised as
  committed (the range is degenerate); wind, mass, and sensor noise/spikes are.

**`severity_anneal_steps` — the disturbance-severity curriculum.** The sampled
ranges are scaled by a severity fraction `min(1, steps / severity_anneal_steps)`
that ramps 0→1 over this many per-env steps (mirroring
`curriculum.anneal_steps`). At severity 0 the episode is nominal; at 1 the full
ranges apply. Pairing this with the task-difficulty anneal lets the agent master
the easy nominal task first, then face progressively harder initial conditions
*and* disturbances together — the fix for wide randomisation destabilising a
cold-start policy. Set to `0` to disable the ramp (full ranges from step 0).
This is the training-time realisation of the `disturbance_severity` axis in
`docs/naming_conventions.md`; it is deliberately distinct from the graduated
matrix's per-cell `disturbance_severity`, which is fixed and seeded rather than
sampled.

---

## `env.curriculum` — task-difficulty scheduler

```yaml
anneal_steps: 1_000_000
schedule: linear        # linear | fixed
task_difficulty: 1.0    # used only when schedule: fixed
```

Task difficulty `d` is computed from the environment's cumulative step count:

```
d = min(1.0,  global_step / anneal_steps)
```

Over the first 1,000,000 environment steps, `d` ramps linearly from 0 to 1,
gradually widening the initial-condition envelope (altitude range, lateral
offset, descent speed). After 1M steps the curriculum holds at full
difficulty. `task_difficulty` scales the initial-condition envelope only — it
does not scale disturbances or adversary weight (see
`docs/naming_conventions.md`). Disturbance magnitude has its own, separate ramp:
`env.domain_randomization.severity_anneal_steps` (above), which anneals the
*disturbance* ranges during training. The two are intended to run together —
harder ICs and harder disturbances phased in on parallel schedules.

The evaluation scripts pin difficulty with `schedule: fixed`, holding
`task_difficulty` at the value set via `eval_pid.task_difficulty` /
`eval_rl.task_difficulty` (default 1.0) so every evaluation run uses the full
envelope regardless of how many training steps have elapsed.

---

## `env.obs_scaler` — observation normalisation bounds

```yaml
position_xy_m: 200.0
position_z_m: 600.0
velocity_mps: 50.0
attitude_rad: 3.14159
angular_rate_rad_s: 6.28318
fuel_mass_kg: 5_000.0
```

`utils/normalisation.py::FixedObsScaler` divides each observation slot by
its bound before passing it to the agent or PID. This puts all signals
roughly in `[−1, 1]`, which is important for neural network training
stability and for the PID's internal unscaling step.

| Scaler bound | Matches physical limit | Consequence if they diverge |
|---|---|---|
| `position_xy_m = 200` | `oob.cylinder_radius_m = 200` | OOB boundary maps to ±1 |
| `position_z_m = 600` | `oob.ceiling_m = 600` | Ceiling maps to −1 |
| `velocity_mps = 50` | Max plausible entry speed | Values above 50 m/s would be clipped at ±1 |
| `attitude_rad = π` | Full rotation | ±180° maps to ±1 |
| `angular_rate_rad_s = 2π` | One full revolution/s | Values above this saturate at ±1 |
| `fuel_mass_kg = 5000` | `initial_fuel_kg` | Full tank → 1.0, empty → 0.0 |

**Important:** if `oob.cylinder_radius_m` or `oob.ceiling_m` are changed,
the matching scaler bounds must be updated to keep the normalisation
consistent. The `position_z_m` scaler is also used for the altitude
component of the dense distance reward.

---

## Parameter interactions summary

```
initial_fuel_kg  ─────────────────────────────┐
dry_mass_kg       → total_mass(t) → TWR(t)    │  determines
max_thrust_N      → u_hover = m·g/T_max       │  limit-cycle
throttle_min      → deadband gap              │  behaviour
                                              ┘

altitude_max_m    ─────────────┐
target_descent_mps             │  must satisfy:
max_steps / control_hz  ───────┘  altitude_max / target_descent < max_steps / hz

oob.cylinder_radius_m = obs_scaler.position_xy_m  (must stay in sync)
oob.ceiling_m         = obs_scaler.position_z_m   (must stay in sync)
```
