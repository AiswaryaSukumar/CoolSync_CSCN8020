# Methodology

## 1. Methodological Scope

This document describes the current implemented CoolSync+ project as it exists
in the repository.

The methodology covers:

- prompt-aware workload abstraction
- prompt-to-heat generation
- thermal environment simulation
- one-step forecasting
- reinforcement learning and baseline control
- dashboard validation

It intentionally reflects the final implementation, not older drafts.

## 2. Problem Decomposition

The cooling-control problem was broken into five technical layers:

1. Prompt abstraction
2. Workload and heat generation
3. Forecasting
4. Thermal control environment
5. Controller evaluation

This decomposition made the project manageable while preserving the full
prompt-prediction-RL direction.

## 3. Prompt-Aware Workload Layer

The project models prompts through structured categories rather than raw NLP.

Prompt classes:

- simple
- reasoning
- code
- long_context
- burst

For each prompt event, the system tracks:

- `prompt_type`
- `prompt_length`
- `complexity_score`
- `concurrency_level`

These values are then converted to workload and heat through
`prompt_model/prompt_to_heat.py`.

This design satisfies the need to connect prompt characteristics to energy and
thermal consequences without expanding the project into a full language-model
deployment study.

## 4. Scenario Framework

The project evaluates control behavior under four scenarios:

- `stable`
- `sinusoidal`
- `spiky`
- `burst_heavy`

The scenarios are defined in `scenarios/scenario_definitions.py` and influence:

- prompt mix
- ambient behavior
- thermal variability

This scenario-based evaluation is important because the value of forecasting is
expected to differ between easy and hard workload conditions.

## 5. Forecasting Method

### 5.1 Data generation

Forecasting data is generated from prompt-driven traces rather than fixed
static datasets. The trace generation pipeline creates sequential heat patterns
that are learnable by an LSTM.

### 5.2 Forecast model

The forecast model is a one-step LSTM implemented in `forecasting/lstm_model.py`.

The primary checkpoint path is:

```text
results/checkpoints/lstm_heat_predictor_best.pth
```

### 5.3 Forecast usage

Forecasting is used in two places:

- in the training/evaluation pipeline through the real checkpoint path
- in the dashboard runtime as part of the env-backed simulation flow

The project therefore supports both research evaluation and interactive
demonstration.

## 6. Thermal Environment

The main environment is `envs/coolsync_env.py`.

Its state includes:

- current temperature
- current workload
- cooling level
- ambient temperature
- previous action
- predicted next-step heat when forecast is enabled

The action space is:

- decrease cooling
- maintain cooling
- increase cooling

The environment also contains:

- nonlinear cooling energy computation
- safe and critical thresholds
- ambient coupling
- step-wise reward computation

## 7. MDP Formulation

The project frames cooling control as a Markov Decision Process.

### State

Conceptually:

```text
s_t = [temperature, workload, cooling_level, ambient_temp, previous_action, predicted_heat]
```

### Action

```text
a_t in {decrease, maintain, increase}
```

### Reward

The reward combines:

- energy penalty
- overheating penalty
- overcooling penalty
- instability penalty
- safe-zone bonus

This creates a multi-objective control problem rather than a single-metric
optimization.

## 8. Controllers

### 8.1 Reinforcement learning

The project includes:

- tabular Q-learning
- DQN

Q-learning is retained as a mathematically transparent RL baseline.
DQN is the stronger function-approximation controller for richer state
representations.

### 8.2 Baselines

The baseline set includes:

- static
- threshold
- PID
- predictive threshold

This comparison ladder is useful because it distinguishes:

- reactive rule-based control
- predictive non-learning control
- predictive learning control

## 9. Dashboard Runtime Method

The dashboard is not a separate toy system anymore. It now uses the core
environment mechanics.

Methodologically, the dashboard route:

1. bridges raw prompt text into structured prompt features
2. runs a 16-step env-backed episode
3. uses the environment's:
   - thresholds
   - energy model
   - reward logic
   - temperature dynamics
4. returns `prompt_features`, `summary`, and `timeseries`

Important caveat:
- the dashboard uses calibrated DQN-like runtime logic
- it does not perform live checkpoint inference from the trained DQN model

That makes the dashboard a faithful demonstration layer, not a full online
deployment of the research model.

## 10. Validation Design

The final validation focused on a compact six-case DQN sweep:

- Stable ON
- Stable OFF
- Spiky ON
- Spiky OFF
- Burst Heavy ON
- Burst Heavy OFF

This sweep was chosen because it demonstrates:

- forecast separation
- thermal resilience differences
- scenario difficulty differences
- energy-versus-safety tradeoffs

## 11. Final Validation Results

| Case | Verdict | Total Energy | Avg Temp | Max Temp | Overheat Count |
|---|---|---:|---:|---:|---:|
| Stable ON | PASS | 7.87 | 25.41 C | 26.11 C | 0 |
| Stable OFF | PASS | 7.54 | 25.76 C | 26.70 C | 0 |
| Spiky ON | PASS | 8.53 | 25.76 C | 26.53 C | 0 |
| Spiky OFF | FAIL | 8.20 | 26.45 C | 27.21 C | 5 |
| Burst Heavy ON | PASS | 9.63 | 26.21 C | 26.99 C | 0 |
| Burst Heavy OFF | FAIL | 9.85 | 26.55 C | 27.48 C | 5 |

## 12. Interpretation of Results

The final methodological conclusion is:

> Forecast-aware control improves resilience under harder thermal conditions,
> sometimes at a modest energy cost.

This is supported by:

- Stable: both modes pass, OFF is cheaper, ON is safer
- Spiky: ON passes while OFF fails
- Burst Heavy: ON passes while OFF fails

This is a strong final demonstration of the value of prediction under difficult
thermal conditions.

## 13. Reproducibility

The project supports reproducibility through:

- configuration in `configs/default_config.py`
- seed control in `utils/seed.py`
- checkpoint saving under `results/checkpoints/`
- logs and summaries under `results/logs/` and `results/summaries/`
- smoke tests under `tests/`

## 14. Reporting Guidance

When writing the final report or slides:

- present the system as a simulation-backed predictive cooling prototype
- keep the distinction between research pipeline and dashboard runtime clear
- use the six-case validation table as the main evidence block
- emphasize thermal resilience under hard scenarios
- treat the modest energy tradeoff as part of the design story, not a failure
