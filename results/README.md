# ZetaBench — Full Results (rocket landing)

The complete evidence behind the headline results in the
[project README](../README.md): per-controller success tables across the task-
difficulty envelope, the graduated disturbance matrix, and the interpretation
with its caveats. Every number here is reproducible from the tracked artifacts
in this folder ([`robustness_matrix.csv`](robustness_matrix.csv),
[`robustness_heatmap.png`](robustness_heatmap.png)) and the commands in the
[usage guide](../docs/usage.md).

> **Protocol.** All three controllers share one evaluation protocol — 100
> episodes per cell, fixed seed (42), 3.0 m/s touchdown gate — so the results
> are directly comparable. PID is a fixed-gain classical baseline (no
> training); SAC and PPO are curriculum-trained to task difficulty 0.3
> (γ=0.999, staged warm-start) under the same reward these tables report.
> Results current as of July 2026.

---

## Landing success vs task difficulty (nominal conditions)

`task_difficulty` linearly anneals the initial-condition envelope from a
near-vertical ~30 m drop (`0.0`) toward a 60 m, 20 m/s, ±50 m off-pad approach
(`1.0`). 100 episodes per cell, seed 42, touchdown threshold 3.0 m/s, 5000 kg
landing-burn reserve.

### SAC

The agent was trained up to task_difficulty 0.3; levels beyond that probe its
zero-shot generalization to harder, unseen approaches.

| Task difficulty [0,1] | Success | Crash | OOB | Touchdown (m/s) | Fuel used (%) | Episode len |
|---|---|---|---|---|---|---|
| 0.0 (near-vertical) | **100%** | 0 | 0 | 2.48 | 13% | 291 |
| 0.1 | **100%** | 0 | 0 | 2.14 | 13% | 282 |
| 0.2 | 99% | 1 | 0 | 2.17 | 13% | 283 |
| 0.3 (train ceiling) | **100%** | 0 | 0 | 1.97 | 13% | 291 |
| 0.4 | **100%** | 0 | 0 | 1.92 | 14% | 296 |
| 0.5 | 98% | 2 | 0 | 1.94 | 14% | 300 |
| 0.6 | 97% | 3 | 0 | 1.90 | 15% | 305 |
| 0.8 | 92% | 8 | 0 | 1.97 | 15% | 314 |
| 1.0 (hardest) | 86% | 14 | 0 | 2.23 | 16% | 325 |

The agent is **~100% within its training envelope (≤0.4)** and degrades
gracefully beyond it — still 97% at 0.6, 92% at 0.8, and 86% at the hardest
full-envelope approach despite never training there. Crucially, **out-of-bounds
is 0 at every level** and touchdown speed stays well under the 3.0 m/s gate
(≤2.5 m/s throughout): the rare failures are gentle soft-braking crashes, never
loss-of-control fly-out.

**Training notes.** Two choices mattered. A discount factor of **γ=0.999** puts
the effective horizon (~1000 steps) beyond typical episode length, keeping the
terminal landing / out-of-bounds reward visible to the optimizer and closing off
a fly-up→out-of-bounds local optimum that otherwise caps success. A staged
curriculum warm-start (fixed difficulty 0.1 → 0.2 → 0.3) then makes the policy
markedly more efficient: at difficulty 0.2, versus a single-stage policy, it
lifts success from 60% to 100% while cutting episode length from ~1180 to ~250
steps and fuel burn from 49% to 12%.

### PPO

PPO was trained with the identical γ=0.999 setting and the same staged
curriculum warm-start (fixed difficulty 0.0 → 0.1 → 0.2 → 0.3), and evaluated
under the same protocol as SAC.

| Task difficulty [0,1] | Success | Crash | OOB | Touchdown (m/s) | Fuel used (%) | Episode len |
|---|---|---|---|---|---|---|
| 0.0 (near-vertical) | **100%** | 0 | 0 | 2.41 | 12% | 266 |
| 0.1 | 99% | 1 | 0 | 2.42 | 12% | 272 |
| 0.2 | **100%** | 0 | 0 | 2.36 | 13% | 281 |
| 0.3 (train ceiling) | **100%** | 0 | 0 | 2.18 | 13% | 290 |
| 0.4 | 99% | 1 | 0 | 2.11 | 14% | 299 |
| 0.5 | **100%** | 0 | 0 | 1.94 | 15% | 310 |
| 0.6 | **100%** | 0 | 0 | 1.82 | 15% | 321 |
| 0.8 | 85% | 3 | 0 | 1.91 | 26% | 588 |
| 1.0 (hardest) | 63% | 16 | 0 | 3.47 | 34% | 792 |

PPO holds **100% through difficulty 0.6** — further out-of-envelope than SAC —
then degrades on the largest approaches. Like SAC it shows **zero
out-of-bounds at every level**; the shortfall at 0.8–1.0 is dominated by
**timeouts on long, fuel-limited descents** (episode length and fuel climb
sharply — 792 steps / 34% fuel at 1.0) rather than loss-of-control.

### PID baseline

The classical PID controller has **fixed gains** — no training, no curriculum —
a single descent-rate loop with an **altitude-scheduled flare** (target descent
= 0.10 · altitude, clamped to [1.0, 8.0] m/s) so it slows progressively toward
the pad. Properly tuned, it lands the **entire envelope at 100%** with soft,
monotonically decreasing touchdown speeds and zero out-of-bounds / zero timeout.
(The flare matters: a constant-target loop cannot bleed enough speed to reach
the gate on a short 30 m drop.)

| Task difficulty [0,1] | Success | Crash | OOB | Touchdown (m/s) | Fuel used (%) | Episode len |
|---|---|---|---|---|---|---|
| 0.0 (near-vertical) | **100%** | 0 | 0 | 1.96 | 14% | 307 |
| 0.1 | **100%** | 0 | 0 | 1.63 | 15% | 339 |
| 0.2 | **100%** | 0 | 0 | 1.39 | 17% | 378 |
| 0.3 | **100%** | 0 | 0 | 1.23 | 19% | 425 |
| 0.4 | **100%** | 0 | 0 | 1.14 | 21% | 477 |
| 0.5 | **100%** | 0 | 0 | 1.08 | 24% | 530 |
| 0.6 | **100%** | 0 | 0 | 1.04 | 26% | 583 |
| 0.8 | **100%** | 0 | 0 | 0.99 | 30% | 684 |
| 1.0 (hardest) | **100%** | 0 | 0 | 0.96 | 34% | 775 |

### PID vs SAC vs PPO (matched conditions)

With an honestly-tuned PID (altitude flare), **PID lands 100% across the whole
envelope**, matching the RL agents in their training region and holding at the
hardest approaches where the RL policies fall off:

- **RL agents (SAC, PPO)** are curriculum-trained up to difficulty 0.3, so they
  are ~100% inside their envelope and degrade on the harder, unseen approaches
  (SAC to 86%, PPO to 63% at 1.0) — with **zero out-of-bounds at every level**;
  failures are gentle soft-braking **crashes** (SAC) or **timeouts** on long
  fuel-limited descents (PPO), never fly-out.
- **PID** with the flare lands the full envelope, and its touchdown speeds
  actually *tighten* with difficulty (taller drops give more runway to settle):
  1.96 m/s at 0.0 down to 0.96 m/s at 1.0.

| Task difficulty [0,1] | PID success | SAC success | PPO success |
|---|---|---|---|
| 0.0 | **100%** | **100%** | **100%** |
| 0.1 | **100%** | **100%** | 99% |
| 0.2 | **100%** | 99% | **100%** |
| 0.3 | **100%** | **100%** | **100%** |
| 0.4 | **100%** | **100%** | 99% |
| 0.5 | **100%** | 98% | **100%** |
| 0.6 | **100%** | 97% | **100%** |
| 0.8 | **100%** | 92% | 85% |
| 1.0 | **100%** | 86% | 63% |

The honest verdict on this vertical-descent task: a **well-tuned classical
controller is a formidable baseline** — RL *matches* it inside the training
envelope but does not beat it, and *trails* it on the hardest approaches. That
is exactly the credibility check ZetaBench exists for — cross-paradigm
comparison on fair, identical conditions, where "RL wins" must be earned, not
assumed. Caveat: `task_difficulty` is **not** a controller-agnostic hardness
axis — for the descent-rate PID, higher difficulty means *taller* drops and thus
*more* settling runway, so its curve is flat where the RL curriculum axis makes
the task harder.

---

## Robustness under graduated disturbances (fair envelope)

> **Protocol.** All three controllers face the graduated **disturbance matrix**
> on identical fixed-seed conditions: the initial-condition envelope is pinned
> at **task difficulty 0.4** (inside the RL agents' training envelope, so the
> comparison is fair) and only disturbance severity varies. PID is the
> flare-tuned baseline above; SAC and PPO are the same curriculum-trained
> agents — all three are **100% at difficulty 0.4 under nominal conditions**
> before disturbances are applied. 100 episodes/cell, seed 42, 3.0 m/s
> touchdown gate. Source: [`robustness_matrix.csv`](robustness_matrix.csv) ·
> heatmap: [`robustness_heatmap.png`](robustness_heatmap.png).

![Robustness heatmap — landing success rate per disturbance type × severity cell, one panel per controller](robustness_heatmap.png)

Mean landing success across the 32-cell disturbance grid:

| Controller | Mean success (32 cells) |
|---|---|
| PID | **81.9%** |
| SAC | 65.4% |
| PPO | 59.7% |

Broken down by disturbance family (mean success across that family's cells):

| Disturbance | PID | SAC | PPO |
|---|---|---|---|
| none (nominal) | **100%** | **100%** | 99% |
| wind (≤ 10 m/s) | **100%** | **100%** | **100%** |
| mass offset (± 20%) | **100%** | **100%** | 29% |
| sensor noise | **56%** | 9% | 18% |
| combined (max) | 0% | 0% | 0% |

**What the matrix shows.**

- **On the physical disturbances, PID and SAC are both perfect.** PID (flare),
  SAC, and PPO all land 100% under nominal and wind; PID and SAC also hold 100%
  under ±20% mass offset. A well-tuned classical controller is fully competitive
  with deep RL here — it does not concede the dynamics-disturbance regime.
- **Sensor noise is the great equalizer.** PID leads (56%) but drops from its
  clean-condition 100%; the RL policies are far more fragile (SAC 9%, PPO 18%) —
  a feed-forward MLP reacting to one raw noisy frame has no temporal filtering.
  An explicit eval-time observation filter *helped* PID but *broke* the RL
  policies, so the gap is architectural, not a training-budget issue.
- **A cross-paradigm split:** PPO collapses under mass offset (29%) where PID and
  SAC are unaffected (100%) — same RL family, very different failure mode.
- **The combined worst-case defeats every controller (0%).**
- **Overall (mean of 32 cells): PID 82%, SAC 65%, PPO 60%.** With an honestly-
  tuned baseline the lead is real, not an artifact — PID owns the physical-
  disturbance rows outright and edges the sensor-noise rows. The credible
  verdict: **no controller is universally robust** (sensor noise and the
  combined cell are open for all), but on this task a well-tuned PID is the one
  to beat.

**Per-controller robustness cards** — degradation curve per disturbance family
plus a break-point severity at a 95% deployment gate — are generated into
[`cards/`](cards/) by `python experiments/robustness_card.py`; robust-trained
variant curves join the same cards as those runs complete.

---

## Interpretation & honest caveats

The headline result is deliberately unflattering to deep RL — and that is the
point of a credible benchmark:

- **A well-tuned classical controller matches or beats RL on this task.** PID
  ties SAC across every physical-disturbance regime and leads overall. RL does
  not "win" here — it *matches inside its training envelope and trails at the
  extremes*.
- **Because the task, as scored, does not require RL-class capability.** Success
  is a soft *vertical* touchdown within 3 m/s; it does **not** require landing on
  the pad, and in fact *no* controller here performs lateral guidance (all three
  touch down ~15–40 m off-target). Reduced to descent-rate regulation, the
  problem is one a tuned PID solves by design.
- **`task_difficulty` is not a controller-agnostic hardness axis.** Higher
  difficulty means taller drops → *more* settling runway for the descent-rate
  PID, so its curve is flat/improving exactly where the RL-curriculum axis is
  meant to get harder. The fixed-difficulty disturbance matrix is the cleaner
  cross-paradigm axis.
- **RL's one clear, structural weakness here is observation noise** — a
  memoryless MLP acting on a raw noisy frame. That is a genuine architectural
  finding, not a training-budget artifact: domain-randomization fine-tuning and
  an eval-time observation filter both failed to close it.

**What this does and doesn't establish.** It establishes that ZetaBench delivers
a *fair, reproducible, cross-paradigm* verdict — here, *"a well-tuned PID is hard
to beat."* It does **not** establish that RL is weak; it shows this task is
under-specified for RL's strengths, and that the RL agents compared here are
*naive* — trained on nominal dynamics, never on the disturbance distribution.

**What comes next (in order).** Rather than replicating this verdict in a second
environment, the next work hardens it in this one:

1. **Naive-vs-robust RL on the same matrix** — retrain SAC/PPO from scratch with
   the existing training-time domain randomisation and re-run the identical
   matrix. This directly tests whether "RL loses because it was not trained for
   these scenarios," which the current results assume but do not test.
2. **Test the sensor-noise "architectural" claim** — a frame-stacked or recurrent
   policy trained with observation noise; either outcome is a finding.
3. **Graduate the combined-disturbance axis** — the max-only combined cell (0%
   for all) says *that* everything breaks, not *at what magnitude*.
4. **Precision (on-pad) landing with 3-axis guidance**, so the task actually
   requires RL/MPC-class capability — **while keeping the classical baseline**
   (extended with a lateral cascade), since a retained, honest baseline is
   exactly what makes any future "RL wins" credible.
5. **LQR/MPC baseline** — the closest analogue to deployed practice (SOCP-style
   descent guidance) and the completion of the "any controller" claim.

The roadmap's contact-rich environments (eVTOL precision landing, bipedal
locomotion) follow after v1.0, once the abstractions they would reuse are
validated by a hardened verdict rather than a confounded one.
