# Reward Engineering Guide For Reinforcement Learning

Reward engineering is the process of designing the feedback signal that teaches a reinforcement learning agent what behavior is desirable. The core rule is simple:

> The agent will optimize the reward you give it, not the behavior you intended.

Reward engineering is therefore mostly about making the reward mathematically aligned with the actual task.

## 1. Define The Task Objective Precisely

Before writing any reward, define what "success" means in measurable terms.

Ask:

1. What is the final goal?
2. What counts as failure?
3. What behaviors are acceptable but suboptimal?
4. What behaviors are dangerous or invalid?
5. Are there constraints the agent must respect?

Example for rocket landing:

- Goal: land upright on the pad.
- Success: touchdown speed below threshold, tilt below threshold, angular velocity low.
- Failure: crash, out of bounds, timeout, fuel depletion.
- Secondary preferences: use less fuel, avoid aggressive control, stay stable.

Do not start with reward weights. Start with outcome definitions.

## 2. Separate Terminal Rewards From Dense Shaping

Most RL tasks benefit from two kinds of reward.

Terminal reward is given at the end of an episode:

```text
+1000 for success
-500 for crash
-300 for out of bounds
-100 for timeout
```

Dense shaping reward is given every step to guide learning:

```text
small reward for moving closer to target
small penalty for high velocity
small penalty for instability
small penalty for wasting energy
```

Terminal rewards define the real objective. Dense rewards help exploration, especially when terminal success is rare.

A common mistake is making dense penalties so large that they dominate the terminal outcome.

## 3. Start With A Simple Baseline Reward

Begin with the smallest reward that encodes the task:

```text
success: +100
failure: -100
otherwise: 0
```

This is sparse and may not learn well, but it gives you a clean reference point. Add shaping only when you can explain what learning problem it solves.

Avoid starting with many weighted terms. That makes debugging much harder.

## 4. Identify The Agent's Available Control Path

Before adding a reward term, ask:

> Can the agent actually influence this quantity through its actions?

Good reward terms are controllable.

- Good: velocity, position error, fuel use, orientation, action smoothness.
- Risky: variables affected mostly by randomness or environment dynamics.
- Bad: rewarding something the agent cannot observe or influence.

If a term is not controllable, it adds noise.

## 5. Use Reward Components With Clear Units

Every reward component should have a clear physical or logical meaning.

Example:

```text
distance_penalty = -0.01 * distance_to_goal_meters
velocity_penalty = -0.05 * speed_mps
fuel_penalty = -0.001 * fuel_used_kg
success_bonus = +500
crash_penalty = -300
```

Then estimate magnitudes.

If an episode is 1000 steps and your dense penalty is around `-1` per step, the agent sees about `-1000` dense reward per episode. A `+100` success bonus will be too small to matter.

The reward scale should match the decision you care about.

## 6. Check For Perverse Incentives

This is the most important reward-engineering step.

Ask:

1. Can the agent get high reward while failing the task?
2. Can it reduce penalty by ending the episode early?
3. Can it exploit a loophole?
4. Does doing nothing produce better reward than trying?
5. Does crashing quickly beat surviving longer?
6. Does the reward punish exploration too harshly?
7. Does the agent get more reward for hovering forever than completing the task?

Example failure:

```text
-0.1 per step
-50 for crash
+100 for success
```

If success is hard, the agent may learn to crash quickly because staying alive accumulates more negative reward. That is not a training bug. That is a reward bug.

## 7. Prefer Progress Rewards Over Raw Penalties

Instead of penalizing absolute distance every step:

```text
reward = -distance_to_goal
```

prefer rewarding improvement:

```text
reward = previous_distance - current_distance
```

Absolute penalties can make long episodes look bad even if the agent is improving. Progress rewards say the agent gets credit when it moves in the right direction.

Examples:

```text
distance_progress = previous_distance - current_distance
velocity_progress = previous_speed - current_speed
orientation_progress = previous_tilt - current_tilt
```

This often reduces the incentive to terminate early.

## 8. Consider Potential-Based Reward Shaping

Potential-based reward shaping is a principled method:

```text
shaping_reward = gamma * Phi(next_state) - Phi(current_state)
```

`Phi(state)` is a potential function measuring how promising a state is.

Example:

```text
Phi(state) =
  - distance_to_goal
  - velocity_error
  - orientation_error
```

This gives dense guidance while preserving the optimal policy under standard assumptions.

For many control problems, this is better than arbitrary per-step penalties. Use it when you want the agent to receive feedback for moving toward better states without accidentally changing the real objective.

## 9. Make Failure Penalties Informative

A flat failure penalty is often too blunt.

Bad:

```text
crash = -100
```

Better:

```text
crash_penalty = -100 - 10 * impact_speed - 50 * tilt_error
```

Now the agent learns that a slower crash is better than a high-speed crash. That creates a gradient toward success.

This is useful when success is initially rare. The agent may not land safely at first, but it can learn to crash slower, crash more upright, touch down near the target, and eventually succeed.

## 10. Avoid Over-Penalizing Necessary Exploration

Action penalties, energy penalties, jerk penalties, and constraint penalties are useful but dangerous.

For example:

```text
fuel_penalty = -1.0 * fuel_used
```

If too large, the agent may learn to do nothing.

Similarly:

```text
action_smoothness_penalty = -large_weight * action_change
```

If too large, the agent may avoid quick corrective actions even when needed.

Regularization terms should usually be much smaller than task-success terms.

> First make the agent solve the task. Then make it elegant.

## 11. Normalize Or Clip Reward Scales

RL algorithms are sensitive to reward scale.

Very large rewards can destabilize learning. Very small rewards can make learning slow.

Common practices:

- Keep most per-step rewards around `[-10, 10]`.
- Keep terminal rewards meaningfully larger but not absurdly huge.
- Clip individual components if they can explode.
- Normalize observations and sometimes rewards.
- Track each reward component separately.

Example:

```text
distance_term = clip(-0.01 * distance, -5, 0)
velocity_term = clip(-0.05 * speed, -5, 0)
terminal_success = +500
terminal_crash = -200
```

Clipping prevents rare extreme states from dominating training.

## 12. Log Reward Components Separately

Never log only total reward.

Log each component:

```text
reward/distance
reward/velocity
reward/fuel
reward/smoothness
reward/terminal
reward/total
episode/success
episode/crash
episode/length
```

This lets you answer:

- Is the agent optimizing the wrong component?
- Is one penalty dominating everything?
- Does reward improve while success stays flat?
- Are successful episodes actually being selected?

A rising reward with zero success is a warning sign.

## 13. Evaluate With The Real Metric, Not Just Reward

Reward is a training signal. It is not always the final metric.

For evaluation, track task metrics directly:

```text
success_rate
crash_rate
timeout_rate
distance_to_goal
impact_speed
fuel_used
episode_length
constraint_violations
```

Your `best_model` should usually be selected by the true task metric, not raw reward.

For example:

- Best landing model: highest success rate.
- Tie-breaker: lowest impact speed or fuel usage.
- Avoid selecting only by highest shaped reward.

This avoids choosing a model that exploited the reward.

## 14. Build A Reward Debugging Loop

Reward engineering is iterative.

A practical loop:

1. Train briefly.
2. Evaluate 20-100 episodes.
3. Watch trajectories.
4. Inspect reward components.
5. Identify the dominant behavior.
6. Adjust one reward idea at a time.
7. Repeat.

Do not tune many weights at once. You will not know what caused the change.

Debugging table:

```text
Behavior observed                  Likely reward issue
------------------------------------------------------------
Crashes quickly                    Living is too costly / crash penalty too small
Does nothing                       Action/fuel penalty too high
Hovers forever                     Success incentive too weak / timeout not penalized
Gets close but fails               Terminal success too sparse / need soft failure gradient
Uses excessive control             Smoothness/fuel regularization too weak
Optimizes reward but not success   Objective and reward misaligned
```

## 15. Use Curriculum If The Task Is Too Hard

If the agent never experiences success, reward shaping may not be enough.

Use curriculum learning:

1. Start with easier initial states.
2. Lower randomness.
3. Loosen success thresholds.
4. Gradually increase difficulty.
5. Evaluate on the full task separately.

Example:

```text
Phase 1: low altitude, no tilt, low velocity
Phase 2: higher altitude, mild lateral offset
Phase 3: full altitude range, larger velocity
Phase 4: full task distribution
```

Do not evaluate only on easy curriculum states. Keep a fixed full-difficulty eval set.

## 16. Use HPO Only After Reward Logic Is Sound

Hyperparameter optimization cannot fix a misaligned reward.

First validate:

- Success rate is nonzero or trending upward.
- Failures are becoming less severe.
- Reward correlates with task quality.
- Best model selection matches the real metric.

Then tune:

```text
learning_rate
gamma
batch_size
entropy coefficient
network size
tau / target update rate
replay buffer size
```

If HPO optimizes reward, make sure reward correlates with success. Better: optimize success rate directly.

## 17. Test The Reward Function

Reward functions deserve tests.

Useful tests:

1. Success beats crash.
2. A slower crash beats a faster crash.
3. Moving closer to the goal improves reward.
4. Crashing quickly does not beat attempting the task.
5. Dense reward over a typical episode does not swamp terminal rewards.
6. Reward values are finite: no NaNs, infinities, or unbounded explosions.

## 18. A Practical Reward Design Template

For a goal-oriented continuous-control problem:

```text
reward =
    potential_based_progress
  + terminal_success_bonus
  + terminal_failure_penalty
  - small_energy_penalty
  - small_action_smoothness_penalty
  - constraint_violation_penalty
```

Where:

```text
potential_based_progress =
    gamma * Phi(next_state) - Phi(current_state)
```

And:

```text
Phi(state) =
  - w_distance * distance_to_goal
  - w_velocity * speed_error
  - w_orientation * orientation_error
```

Terminal example:

```text
if success:
    reward += success_bonus

if failure:
    reward -= base_failure_penalty
    reward -= impact_or_error_scaled_penalty
```

This tends to work better than raw accumulated penalties.

## 19. Recommended Workflow

A strong general workflow is:

1. Define success and failure metrics.
2. Implement sparse terminal reward.
3. Add simple progress shaping.
4. Log all reward components.
5. Train briefly.
6. Watch actual trajectories.
7. Check whether reward correlates with success.
8. Fix loopholes.
9. Add regularizers only after task learning begins.
10. Change model selection to the real metric.
11. Run longer training.
12. Only then run HPO.

## 20. Key Principles To Remember

- Reward is a specification, not just a signal.
- The agent exploits math, not intent.
- Dense shaping should guide, not dominate.
- Terminal success should be clearly better than any failure.
- Failure penalties should contain useful gradients.
- Log components separately.
- Select best models by task metrics.
- Tune hyperparameters only after reward alignment is correct.

## Note For This Project

For the rocket-landing case in this repository, the reward has moved through the
priorities above and now implements them (see `configs/reward.yaml` and
`envs/reward.py`):

- **Potential-based dense shaping.** The dense term is `Φ`-based rather than a raw
  penalty sum, so shaping guides without changing the optimal policy (PBRS
  invariance). Per-term weights live in `reward.potential`.
- **Impact-aware terminal outcome.** The crash penalty is no longer flat — it
  scales with touchdown speed, tilt, angular rate, and lateral error
  (`reward.terminal.touchdown_*_weight`), with an enforced outcome ordering
  (safe landing > slow upright crash > fast tilted crash > out-of-bounds).
- **Model selection by task metric.** `best_model` is chosen on success rate, not
  mean shaped reward.

Two recent, related tightenings target a residual failure mode — policies that
arrive nearly stopped but still a few m/s too fast to count as a landing:

- **Terminal touchdown-speed penalty strengthened.**
  `reward.terminal.touchdown_speed_weight` was raised **30 → 45** (× normalised
  touchdown speed²), making a fast arrival distinctly worse than a slow one at the
  moment of impact.
- **Near-pad landing-speed shaping added.** A dense speed penalty
  `reward.potential.landing_speed_weight` (currently `4.0`) is gated to the final
  approach by `gate = exp(-altitude / ground_gate_altitude_m)` (≈1 at touchdown,
  decaying with altitude). It pushes the agent to bleed off speed during the flare
  rather than only at the terminal step. Because the gate multiplies a term that is
  zero at the landed (zero-speed) state, the potential optimum is unchanged and the
  shaping stays PBRS-safe. The complementary vertical/lateral velocity weights were
  also increased (`velocity_weight 1.0→1.5`, `vertical_velocity_weight 1.5→3.0`).

Together with the touchdown speed threshold (`env.touchdown.velocity_threshold_mps
= 3.0 m/s`), these push the policy toward the softest touchdown it can achieve while
keeping success — not merely inside — the gate.