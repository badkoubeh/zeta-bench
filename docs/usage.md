# ZetaBench — Usage Guide

Everything you need to install, train, evaluate, render, and configure
ZetaBench. The [README](../README.md) is the project overview; this guide is
the operator manual. Full benchmark results live in
[`results/README.md`](../results/README.md).

---

## Getting Started

### Prerequisites

- **Python ≥ 3.12** and **git**.
- **No system ffmpeg required** — the MP4 renderer uses the binary bundled with
  `imageio-ffmpeg`.

### 1. Set up the environment

This project is configured via `pyproject.toml` (with `[dev]` / `[train]` extras).
The recommended setup uses [uv](https://docs.astral.sh/uv/):

```bash
cd zeta-bench
uv venv --python 3.12            # create .venv with Python 3.12
source .venv/bin/activate

# Pick the extras for what you want to do:
uv pip install -e ".[dev]"          # PID eval + tests only — no torch needed
uv pip install -e ".[dev,train]"    # + RL training stack (torch + SB3) — Linux / Apple Silicon
uv pip install -e ".[dev,cloud]"    # + SageMaker launch SDK (for cloud fan-out only)
```

| Extra | Pulls in | Install it when you want to… |
| --- | --- | --- |
| `dev` | pytest, ruff, mypy, pre-commit | run the PID baseline, tests, and lint |
| `train` | torch, Stable-Baselines3, Optuna sweeper | train SAC/PPO or run HPO (Linux or Apple Silicon) |
| `cloud` | sagemaker, boto3 | launch SageMaker Training Jobs from your laptop |

> **Intel (x86) Mac:** PyTorch dropped x86 macOS wheels, so `[train]` won't install
> there — use the PID path locally, or train on Apple Silicon, Linux, or in Docker.

> `requirements.lock` is a pinned Linux / py3.12 lockfile used by Docker and CI.
> To reproduce that exact dependency set on Linux: `uv pip sync requirements.lock`.
> The `uv pip install -e ".[dev]"` line above is the cross-platform dev path.

<details>
<summary>No <code>uv</code>? Use the stdlib venv + pip</summary>

```bash
cd zeta-bench
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```
</details>

### 2. Run the working experiment — PID baseline eval

The PID baseline is the fully implemented end-to-end path. It builds the
environment, flies the cascaded PID controller, and writes per-episode metrics.

```bash
python experiments/evaluate_pid.py                                  # 20 episodes, full envelope
python experiments/evaluate_pid.py seed=7 eval_pid.n_episodes=20    # more episodes, different seed
python experiments/evaluate_pid.py eval_pid.task_difficulty=0.0     # easiest envelope (low drop, no lateral offset)
make eval-pid SEED=42                                               # Makefile shortcut (runs locally)
```

### 3. Train an agent (SAC / PPO)

Training is config-driven: a **compute profile** picks the device and batch sizes, an
**agent** picks the algorithm, and everything else is an override. Pick the block that
matches your hardware. Outputs (including `best_model.zip`) land in `results/{run_name}/`.

**MacBook (Apple Silicon / M-series, MPS):**

```bash
# Quick run on the Metal GPU; the fallback flag covers any op MPS doesn't support yet.
PYTORCH_ENABLE_MPS_FALLBACK=1 python experiments/train.py compute=mps agent=sac

# Coarse hyperparameter search sized for a laptop (MPS, ~400k steps, 10 trials):
PYTORCH_ENABLE_MPS_FALLBACK=1 python experiments/train.py -m \
    --config-name=hpo_sac budget=laptop
```

**CUDA GPU (workstation, or a SageMaker Studio / cloud notebook):**

```bash
python experiments/train.py compute=large_gpu agent=sac total_steps=2000000
python experiments/train.py compute=large_gpu agent=ppo total_steps=2000000

# Full Optuna HPO sweep (20 trials, 2M steps each):
python experiments/train.py -m --config-name=hpo_sac compute=large_gpu budget=full
```

This is exactly what you run inside a SageMaker Studio JupyterLab space on a GPU instance
(e.g. `ml.g5.xlarge`) after `pip install -e ".[train]"` and (optionally)
`export WANDB_API_KEY=...`. Device resolution falls back to CPU if the space has no GPU,
so the same command is safe on a CPU instance.

**Train under domain randomisation (optional).** By default training runs on nominal
dynamics. To harden a policy across the disturbance distribution, enable the training-time
domain-randomisation wrapper — every training episode then draws a fresh wind / mass /
sensor-noise disturbance from `configs/env.yaml::env.domain_randomization`:

```bash
python experiments/train.py compute=large_gpu agent=ppo \
    env.domain_randomization.enabled=true \
    env.domain_randomization.severity_anneal_steps=500000
```

It wraps the **training vec-env only** — the eval / model-selection env and the graduated
robustness matrix stay nominal and deterministic, so model selection and cross-controller
comparison are unaffected. `severity_anneal_steps` ramps the disturbance magnitudes 0→1
alongside the task-difficulty curriculum so a cold-start policy masters the easy nominal
task before facing wide randomisation. See `docs/env_config_reference.md` for the full
knob list.

**Progressive profile (Stage A → gate → Stage B).** One command chains a nominal
curriculum stage, a verification gate at the training envelope, and a
domain-randomised hardening stage per agent:

```bash
python experiments/train_profile.py profile=progressive          # SAC + PPO, full budgets
python experiments/train_profile.py profile=smoke                # minutes-scale wiring check
python experiments/train_profile.py profile=progressive "profile.agents=[sac]"
```

**Amazon SageMaker (managed jobs — parallel seed / HPO fan-out):**

> **Honest note:** SB3's SAC is off-policy and single-process — a multi-GPU/multi-node
> job does *not* speed up a single run. The effective use of a fleet is many independent
> **single-GPU** jobs (one seed or HPO trial each), which the launcher below does. The
> `multi_gpu` profile's `data_parallel` flag is a placeholder and is not yet honored.

Build and push the purpose-built `sagemaker` image stage, then fan out jobs with the
SageMaker SDK (`pip install -e ".[cloud]"`):

```bash
# Build the SageMaker-target image (add --platform linux/amd64 on Apple Silicon)
docker build --target sagemaker -t <account>.dkr.ecr.<region>.amazonaws.com/zeta-bench:sm .
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker push <account>.dkr.ecr.<region>.amazonaws.com/zeta-bench:sm

# Fan out 3 independent single-GPU jobs, one per seed
python experiments/sagemaker_launch.py seeds \
    --image-uri <account>.dkr.ecr.<region>.amazonaws.com/zeta-bench:sm \
    --role arn:aws:iam::<account>:role/SageMakerExecutionRole \
    --s3-output s3://my-bucket/zetabench/ \
    --seeds 0 1 2 --total-steps 2000000

# Or a Bayesian HPO sweep (12 trials, 4 concurrent)
python experiments/sagemaker_launch.py hpo \
    --image-uri ...:sm --role ... --s3-output s3://my-bucket/zetabench/ \
    --max-jobs 12 --max-parallel-jobs 4
```

`docker/sm-entrypoint.sh` maps SageMaker conventions onto the Hydra entrypoint:
hyperparameters become CLI overrides, training writes to `/opt/ml/checkpoints`
(continuously synced to S3 for spot-instance resumability), and the final model lands in
`/opt/ml/model` → `model.tar.gz` in your `--s3-output`. Pass your WandB key via the
launching environment (e.g. AWS Secrets Manager); it is forwarded to each job, never committed.

**CPU (dev / smoke test):**

```bash
python experiments/train.py compute=cpu total_steps=20000        # short run to verify wiring
make train COMPUTE=cpu AGENT=sac SEED=42                          # same, inside Docker
```

Resume an interrupted run with `resume_from=results/<run_name>/<checkpoint>.zip`. If a
requested accelerator is unavailable the agent logs a warning and falls back to CPU, so
the same command is safe anywhere. (`train_mode=adversarial` is not wired yet — see below.)

### 4. Evaluate a trained agent

`evaluate_rl.py` flies a trained SAC/PPO policy over the full envelope and writes the same
per-episode metrics as the PID path. Point it at a local checkpoint or a W&B artifact:

```bash
# From a local checkpoint:
python experiments/evaluate_rl.py agent=sac \
    eval_rl.model_path=results/sac_moderate_nominal_42/best_model.zip

# From the W&B model registry:
python experiments/evaluate_rl.py agent=sac \
    eval_rl.model_artifact="entity/project/zetabench-sac:best"

# Add rendering (best/worst-episode plots + MP4):
python experiments/evaluate_rl.py agent=ppo \
    eval_rl.model_path=results/ppo_moderate_nominal_42/best_model.zip \
    eval_rl.render=true
```

Local checkpoint evaluations write next to the checkpoint by default, e.g.
`results/sac_moderate_nominal_42/eval_rl_p1_seed42/summary.json`. Override
`results_dir=...` when you want a custom output location.

### 5. Run the robustness matrix and cards

```bash
# The graduated disturbance matrix (all controllers; RL checkpoints load from results/)
python experiments/evaluate_robustness.py

# Per-controller robustness cards (degradation curves + break-points) from the matrix CSVs
python experiments/robustness_card.py
python experiments/robustness_card.py robustness_card.gate=0.90
```

Matrix outputs land in `results/` (`robustness_matrix.csv`, `robustness_heatmap.png`);
cards land in `results/cards/{controller}.{png,json}`.

### 6. Render videos and plots

Both eval entry points share a `render` toggle (off by default). For the PID baseline:

```bash
python experiments/evaluate_pid.py eval_pid.render=true eval_pid.render_fps=50
make viz SEED=42                                                    # shortcut for the above
```

For a trained agent, use `eval_rl.render=true` (see step 4).

Outputs land in `results/{run_name}/`, where `run_name = pid_moderate_eval_{seed}`:

```
results/pid_moderate_eval_42/
├── episodes.csv                       # one row per episode (outcome, return, touchdown speed, fuel)
├── summary.json                       # aggregate stats (success rate, means, ...)
├── plots/timeseries_ep{idx}_{outcome}.png   # best/worst episode time series (render=true)
└── video/landing_ep{idx}_{outcome}.mp4      # 2D side-view animation        (render=true)
```

Rendering is **off by default** so the tune-and-rerun loop stays fast — MP4
generation is the slow step.

### 7. Run the tests

```bash
pytest tests/ -v                 # full suite (enforces a 90% coverage gate)
pytest tests/test_physics.py -v  # physics invariants only
```

Coverage is configured in `pyproject.toml` and runs automatically: a **90%
branch-coverage gate** spanning `dynamics`, `envs`, `controllers`, and `utils`,
with an HTML report written to `htmlcov/`.

Tests cover:
- Energy conservation across dynamics integration
- Thrust vector bounds and normalisation
- Observation space shape and bounds
- Reward range sanity
- Agent checkpoint save/load

---

## Config quick reference

The knobs you'll actually reach for, by task. Everything is a Hydra override appended to
the command (`key=value`), and `-m` turns a run into a sweep. The full parameter set lives
in `configs/` — these are the common ones.

**Where am I running it? (`compute=`, `budget=`)**

| Override | Options | Use it to… |
| --- | --- | --- |
| `compute=` | `cpu`, `mps`, `small_gpu`, `large_gpu`, `kaggle_gpu` | pick device + batch/buffer sizes for your hardware (MacBook → `mps`, cloud GPU → `large_gpu`) |
| `budget=` | `laptop`, `full` | HPO only: laptop = ~400k steps / 10 trials (implies `compute=mps`); full = 2M steps / 20 trials |

**What am I running? (`agent=`, `--config-name=`)**

| Override | Options | Use it to… |
| --- | --- | --- |
| `agent=` | `sac`, `ppo`, `pid` | choose the controller/algorithm |
| `--config-name=` | `train`, `hpo_sac`, `hpo_ppo` | switch from a single run to an Optuna sweep (pair with `-m`) |
| `train_mode=` | `nominal` | nominal works today; `adversarial` is not yet wired (see below) |

**How big / how reproducible? (scale + seeds)**

| Override | Default | Use it to… |
| --- | --- | --- |
| `total_steps=` | `2000000` | set training length (lower for smoke tests) |
| `seed=` | `42` | fix the seed for reproducible runs |
| `eval_callback.every_n_steps=` | `50000` | how often to evaluate for best-model selection |
| `eval_callback.n_eval_episodes=` | `20` | episodes per evaluation |

**Harden against disturbances? (`env.domain_randomization.*`, training only)**

| Override | Default | Use it to… |
| --- | --- | --- |
| `env.domain_randomization.enabled=` | `false` | randomise wind / mass / sensor noise per training episode (eval + robustness matrix stay nominal) |
| `env.domain_randomization.severity_anneal_steps=` | `0` | ramp disturbance magnitude 0→1 over N per-env steps (`0` = full ranges from step 0) |

**Output features (rendering, episodes, difficulty)**

| Override | Default | Use it to… |
| --- | --- | --- |
| `eval_pid.render=` / `eval_rl.render=` | `false` | write best/worst-episode PNG plots + MP4 video |
| `eval_pid.render_fps=` / `eval_rl.render_fps=` | `50` | set rendered-video frame rate |
| `eval_pid.n_episodes=` / `eval_rl.n_episodes=` | `20` / `100` | number of evaluation episodes |
| `eval_pid.task_difficulty=` / `eval_rl.task_difficulty=` | `1.0` | pin task difficulty (`0.0` easiest … `1.0` full envelope) |
| `eval_rl.model_path=` | `null` | evaluate a local checkpoint `.zip` |
| `eval_rl.model_artifact=` | `null` | evaluate a checkpoint pulled from the W&B registry |

> **Adversarial mode not yet runnable.** Adversarial training
> (`train_mode=adversarial`) still raises `NotImplementedError`. The graduated
> robustness sweep (`python experiments/evaluate_robustness.py`) runs today.

All results, checkpoints, and videos are saved to `results/{run_name}/`.

---

## Configuration

All hyperparameters are config-driven (Hydra) — nothing is hardcoded. Override
any parameter at the command line:

```bash
# PID eval:
python experiments/evaluate_pid.py eval_pid.n_episodes=20
python experiments/evaluate_pid.py eval_pid.task_difficulty=0.5
python experiments/evaluate_pid.py seed=123

# Training (any dotted path is overridable):
python experiments/train.py env.dynamics.fidelity=high
python experiments/train.py agent.learning_rate=3e-4
python experiments/train.py env.curriculum.anneal_steps=500000
```

Config files:
- `configs/train.yaml` — top-level training composition
- `configs/eval_pid.yaml` — PID baseline eval composition
- `configs/eval_rl.yaml` — trained-agent eval composition
- `configs/eval_robustness.yaml` — robustness-matrix eval composition
- `configs/robustness_card.yaml` — per-controller robustness cards
- `configs/env.yaml` — environment, dynamics, and training-time domain-randomisation parameters
- `configs/reward.yaml` — all reward weights (potential-based dense shaping + impact-aware terminal)
- `configs/pid_controller.yaml` — PID gains
- `configs/adversary.yaml` — adversary hyperparameters (adversarial mode not yet wired)
- `configs/agent/{sac,ppo,pid}.yaml` — per-algorithm hyperparameters
- `configs/compute/{cpu,mps,small_gpu,large_gpu,kaggle_gpu}.yaml` — device profiles
- `configs/budget/{laptop,full}.yaml` — HPO sweep budgets
- `configs/profile/{progressive,smoke}.yaml` — staged training-profile budgets

---

## Experiment Tracking

All runs are tracked in [Weights & Biases](https://wandb.ai).

### WandB setup

Create a `.env` file in the repo root with your personal API key:

```bash
cp .env.example .env          # start from the template
# then open .env and paste your key from https://wandb.ai/authorize
```

`.env` is git-ignored — it never leaves your machine. The training and
evaluation scripts load it automatically via `python-dotenv`.

**Online vs. offline is automatic.** The WandB mode is resolved from the
environment:

- `WANDB_MODE` set explicitly → that value wins (e.g. `WANDB_MODE=offline` to
  force-disable logging even when a key is present).
- otherwise, **`online` when a `WANDB_API_KEY` is available, `offline` when it is
  not** — so simply providing a key turns tracking on, and runs without one never
  block on a login prompt.

> **Team / CI use:** set `WANDB_API_KEY` as an environment variable or a
> GitHub Actions Secret instead of a `.env` file. The same key name is used.

```bash
# Logged automatically per run:
# - All reward components (separately, not just total)
# - Curriculum difficulty level
# - Agent + adversary losses
# - Landing success rate, touchdown velocity, fuel consumption
# - Full robustness matrix as wandb Table
```

---

## Upgrade Paths

| Upgrade | Effort | What changes |
|---|---|---|
| High fidelity dynamics | Medium | New `HighFidelityDynamics` subclass + obs extension |
| Transformer policy | Medium | Swap MLP backbone in SAC/PPO |
| eVTOL / UAV environment | Medium | New dynamics class + new env wrapper |
| CARLA simulator | High | Replace Gymnasium env; rest unchanged |
