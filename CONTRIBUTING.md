# Contributing — ZetaBench

Development guidelines and architecture decisions for this project.
Read this before contributing changes, suggestions, or additions.

---

## How to Contribute (humans and AI agents)

This is an open-source project. Contributions are welcome from human
developers and from AI coding agents alike — **both follow the same standards
described in this document.** An AI coding agent operating on this repository
acts as a contributor and is held to the guidelines here, no differently from
a human contributor. Tooling choices are private to each contributor and must
not leak into the project (see *Commits and pull requests* below).

### Workflow

1. Work on a branch — never commit directly to `main`.
2. Keep changes scoped, and keep the import chain intact (see
   *Module Dependency Rules*).
3. Add or update tests for every change (see *Testing*). The suite must pass
   and coverage must stay at or above the configured threshold.
4. If you change signal flow, the observation/action/adversary spaces, the
   dynamics interface, or the reward structure, update the README diagram
   (see *Diagram Maintenance*) — the pre-commit hook enforces this locally.
5. Never hardcode values that belong in `configs/`.

### Commits and pull requests

- Write clear, imperative commit messages describing *what* changed and *why*.
- **Keep the repository tool-agnostic.** Do not add AI-tool branding,
  "Generated with …" notes, or `Co-Authored-By` trailers naming an AI tool to
  commits, code, or docs. Commit under your own identity.
- Open a pull request against `main`; CI runs tests, coverage, and the diagram
  check.

### For AI coding agents specifically

- Treat this file as your contract: read it fully before editing anything.
- Do not introduce tool-specific files into the tracked repo. Keep any
  agent/editor configuration under git-ignored paths (e.g. `.claude/`).
- Refer to the project owner as "the maintainer" or "a contributor" in code
  and docs — never as "the user".
- Surface assumptions, trade-offs, and limitations honestly (see
  *Known Limitations*).

---

## Project Summary

**ZetaBench** is an open-source, physics-grounded environment where any controller —
deep RL, MPC, LQR, PID, or world-model-based — is stress-tested for robustness across a
graduated disturbance matrix, producing reproducible, comparable evidence of how and
where each controller fails. The framework is domain-agnostic by design (see *Adding an
environment* below); the **current reference environment is 6-DOF rocket landing**, which
exercises the full stack:

- Physics derived from first principles (not a tutorial copy)
- Graduated, fixed-seed disturbance matrix — every controller faces identical conditions
- Cross-paradigm controller comparison: PID, LQR (planned), MPC (planned), SAC, PPO
- Reproducible robustness heatmap: disturbance type × magnitude × success rate

The architecture decisions below describe the rocket-landing reference
environment. Additional environments (e.g. eVTOL/UAV landing, bipedal locomotion) are a
roadmap item, not yet implemented.

---

**Write code at this level.** No tutorial-quality implementations. No hand-waving
on physics. Assume deep familiarity with control theory, RL, and production ML.

---

## Architecture Decisions (DO NOT change without discussion)

### Dynamics
- Abstract base class `RocketDynamics` with `step()` and `get_params()`
- `ModerateFidelityDynamics` is the current implementation (6-DOF rigid body)
- High fidelity upgrade path exists via config flag — do not break this
- `dynamics/` is self-contained. Only `envs/` may import from it

### Observation Space (17-dim)
`[x, y, z, vx, vy, vz, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate, Tx, Ty, Tz, fuel_mass, fuel_remaining]`

### Action Space
Continuous 3D thrust vector `[Tx, Ty, Tz]` ∈ `[-1, 1]³`

### Controllers
All controllers implement the same interface so they can be evaluated on identical
conditions. Current: PID baseline, SAC, PPO. Planned: LQR, MPC. Do not remove any
existing controller; the cross-paradigm comparison is the mechanism that makes the
robustness verdict credible.

### Robustness Strategy
Primary: graduated, fixed-seed disturbance matrix (`robustness/evaluation.py`) — every
controller faces identical conditions, results are directly comparable. Optional power
feature: adversarial/worst-case disturbance search (`robustness/adversarial.py`) — finds
the disturbance that breaks a given controller but is NOT used for cross-controller
comparison (an adaptive adversary fights each controller differently). Report adversarial
findings separately. Disturbance types: wind force, mass uncertainty, sensor noise,
actuator delay.

### Reward
Hybrid: dense shaping every step + sparse terminal bonus/penalty.
All coefficients in `configs/reward.yaml` — never hardcode reward weights.

### Curriculum
Automatic **task-difficulty** annealing: `task_difficulty ∈ [0, 1]` ramps via a
config-driven scheduler (`env.curriculum.schedule: linear | fixed`). It widens the
**initial-condition envelope only** (drop height, lateral offset, descent speed) — it does
NOT scale disturbances or adversary weight (those are the separate `disturbance_severity`
and adversarial axes). See "Naming conventions" below.

### Naming conventions (difficulty vs. severity vs. long tail)
Keep these axes distinct in code, configs, and prose (prior art in parentheses):
- **`task_difficulty ∈ [0, 1]`** — hardness of the *nominal* task (the initial-condition
  envelope); the training curriculum scalar, initial conditions only (legged_gym / Isaac Lab).
- **`disturbance_severity`** — *magnitude* of an external disturbance, graduated on the
  comparable matrix; per perturbation target — exogenous force / state / action / parameter
  (Hendrycks ImageNet-C `severity`; safe-control-gym; realworldrl_suite).
- **`rare_event`** — long-tail / "curse of rarity": rare, extreme events keyed by occurrence
  *probability* + extreme magnitude — a SEPARATE axis from severity, not its maximum.
- **adversarial** — learned worst-case disturbance; reported separately, never in the
  comparable matrix.

`task_difficulty` is implemented; `disturbance_severity` and `rare_event` are reserved
vocabulary for the (not-yet-built) disturbance matrix. Do not reuse "difficulty" for
disturbances; `fidelity` is a modeling choice, not a difficulty.

---

## Adding an environment (roadmap)

ZetaBench ships one reference environment today (rocket landing). New environments
are welcome and should plug into the existing layering rather than fork it. The
extension surface a new environment builds against:

- **Dynamics** — implement the `RocketDynamics`-style contract in `dynamics/`
  (`step()` + `get_params()`). A domain-neutral `BaseDynamics` split is a planned
  refactor; until then, follow the existing abstract base.
- **Environment** — expose a standard Gymnasium `gym.Env` in `envs/` and register
  it with a versioned id (the rocket env registers `RocketLanding-v0`).
- **Reward / curriculum** — keep all coefficients in `configs/`; never hardcode.
- **Controllers** — `SACAgent` / `PPOAgent` are domain-agnostic and adapt to any
  observation/action space; only domain-specific baselines (like the rocket PID)
  need new code.
- **Robustness** — disturbance types in `robustness/disturbances.py` should be
  physically meaningful for the new domain; the evaluation matrix in
  `robustness/evaluation.py` must treat all controllers fairly on identical seeds.

If you are planning a second environment, open an issue first so we can align on
the shared abstractions (env registry, domain config group) before the code lands.

---

## Diagram Maintenance

The closed-loop control diagram in `README.md` (under `## Architecture`) is the
authoritative visual specification of the system topology. **Update both the
diagram and any affected dimensions/units in the surrounding text whenever you
change:**

1. Action space — dimensions, semantics, or scaling
2. Observation space — dimensions, slot semantics, or normalisation bounds
3. Adversary action or observation space
4. Dynamics interface — `RocketDynamics.step()` signature, fidelity tiers
5. Reward decomposition — dense vs sparse vs terminal structure
6. Modules inserted into the loop — world model, recurrent policy, additional sensors

Trigger paths (the `pre-commit` hook at `scripts/check_diagram_sync.py`
enforces this locally; the `.github/workflows/diagram-check.yml` workflow
surfaces a soft warning on pull requests):

- `dynamics/**/*.py`
- `envs/rocket_landing_env.py`, `envs/__init__.py`
- `controllers/**/*.py`
- `adversary/**/*.py`
- `configs/{env,reward,adversary}.yaml`

When in doubt: would a reviewer reading **only the README diagram** get the
right mental model of what was just changed? If not, update the diagram.

---

## Coding Standards (enforce strictly)

```python
# Type hints on all functions
def step(self, state: State, action: Action, dt: float) -> State:

# NumPy-style docstrings with units
def compute_thrust(self, throttle: float) -> np.ndarray:
    """
    Compute thrust vector from normalised throttle command.

    Parameters
    ----------
    throttle : float
        Normalised throttle in [-1, 1].

    Returns
    -------
    np.ndarray
        Thrust vector [Tx, Ty, Tz] in Newtons.
    """
```

- No hardcoded values — everything in `configs/`
- No `print()` for logging — use Python `logging` module
- All random ops use seeded `np.random.Generator`, never `np.random.seed()`
- Units must be documented in docstrings (SI units throughout)
- Tests for every physics function in `tests/`

---

## Config System

Hydra-managed YAML configs. Entry point:

```bash
python experiments/train.py --config-name train
python experiments/train.py dynamics.fidelity=high        # override
python experiments/train.py agent=ppo                     # swap agent
```

Config files:
- `configs/train.yaml` — top-level, composes others
- `configs/env.yaml` — environment + dynamics params
- `configs/reward.yaml` — all reward weights
- `configs/adversary.yaml` — adversary hyperparams
- `configs/agent/sac.yaml` — SAC hyperparams
- `configs/agent/ppo.yaml` — PPO hyperparams

---

## Experiment Tracking

All runs logged to wandb. Required logged values:
- Every reward component separately (not just total reward)
- Curriculum task difficulty (current annealed level; `curriculum/task_difficulty`)
- Adversary loss alongside agent loss
- Episode metrics: landing success, touchdown velocity, fuel used
- Evaluation metrics: robustness matrix results

wandb project name: `zetabench` (legacy runs used `zeta-bench`; new runs use `zetabench`)
Run naming convention: `{agent}_{fidelity}_{adversarial|nominal}_{seed}`
Example: `sac_moderate_nominal_42`

---

## Module Dependency Rules

```
configs → dynamics → envs → controllers → robustness → experiments
```

**Never** import upward in this chain.
**Never** import from `experiments/` in any other module.
`experiments/` contains entrypoints only — no business logic.

---

## Testing

```bash
pytest tests/ -v                         # run all
pytest tests/test_physics.py -v         # physics correctness only
pytest tests/ --cov=dynamics --cov=envs  # coverage
```

Required test coverage:
- `dynamics/`: energy conservation, thrust bounds, mass depletion
- `envs/`: observation shape, action clipping, reward range
- `controllers/`: PID output bounds, agent load/save

---

## Reproduction

Single command to reproduce any result:

```bash
python experiments/train.py --config-name train seed=42
```

All results, checkpoints, and videos saved to `results/{run_name}/`.

---

## Robustness Evaluation

```bash
python experiments/evaluate_robustness.py checkpoint=results/sac_moderate_adversarial_42/
```

Outputs:
- `results/robustness_matrix.csv` — full disturbance sweep table
- `results/side_by_side.mp4` — naive vs robust agent video
- wandb table logged automatically

---

## Current Phase

> Update this section as phases complete.

- [x] Phase 1 — Foundation (Days 1–4) — Track A infrastructure + Track B physics core complete; `notebooks/physics_derivation.ipynb` deferred (maintainer-authored)
- [ ] Phase 2 — Controllers (Days 5–10) — interchangeable controller interface; PID, SAC, PPO evaluated on identical conditions via `robustness/evaluation.py`
- [ ] Phase 3 — Robustness Matrix (Days 11–18) — graduated disturbance matrix, signature heatmap, optional adversarial mode as separate power feature
- [ ] Phase 4 — Polish & OSS Release (Days 19–21)

---

## Renaming: ZetaRL → ZetaBench

The project is being renamed. New code and docs use **ZetaBench** / `zetabench` from the
start. When migrating existing references, do it in scoped commits (one for package/imports,
one for configs, one for docs/CI) rather than a single repo-wide find-replace. See the
full migration order in `.claude/CLAUDE.md` under "Renaming ZetaRL → ZetaBench."

---

## Known Limitations (be honest)

Document in `README.md` as work progresses:
- Adversarial training may be unstable — fallback is domain randomisation
- Moderate fidelity omits aerodynamics and gimbal actuator dynamics
- Training is CPU/single-GPU; no distributed training
- No real hardware validation
