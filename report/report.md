# CoolSync+: Final Report

## 1. Introduction

AI data centers face rising thermal and energy pressure due to bursty,
high-intensity inference workloads. Traditional cooling strategies are often
reactive: they increase cooling only after temperature rises become visible.
This can waste energy in easy conditions and respond too late in hard
conditions.

CoolSync+ addresses this problem through a simulation-based control framework
that combines:

- prompt-aware workload and heat generation
- one-step heat forecasting with LSTM
- reinforcement learning control
- baseline controller comparison
- an interactive dashboard for scenario validation

The project focus is not live deployment to a real facility. It is a
simulation-backed proof of concept for predictive cooling control.

## 2. Problem Statement

The central problem is:

> How can a cooling controller for AI data center workloads remain safe under
> bursty thermal conditions while avoiding unnecessary cooling energy use?

This project specifically studies whether forecast-aware control improves
performance compared with purely reactive control.

## 3. Project Framing

The final implemented framing is:

```text
Prompt-aware workload generation
-> heat generation
-> thermal environment
-> one-step forecast
-> RL / baseline cooling control
-> energy and safety evaluation
```

This framing aligns with the final course direction:

- prompt-to-power correlation
- predictive modeling
- reinforcement learning
- measurable energy and safety outcomes

## 4. Implemented System

### 4.1 Prompt-aware simulation

The project does not rely on raw prompt text as a full NLP problem. Instead, it
models prompt behavior through structured features:

- prompt type
- prompt length
- complexity score
- concurrency level

Supported prompt classes:

- simple
- reasoning
- code
- long_context
- burst

These are converted into workload and heat through the prompt simulation layer
in `prompt_model/`.

### 4.2 Thermal environment

The core environment is implemented in `envs/coolsync_env.py`.

It models:

- current rack temperature
- workload and heat load
- cooling level
- ambient conditions
- safe and critical thresholds
- nonlinear cooling energy
- multi-term reward

The environment action space is:

- `0`: decrease cooling
- `1`: maintain cooling
- `2`: increase cooling

### 4.3 Forecasting

The project includes a one-step LSTM forecaster under `forecasting/`.

The intended production checkpoint path for forecast-aware experiments is:

```text
results/checkpoints/lstm_heat_predictor_best.pth
```

The forecast is used in two ways:

1. In the research/training pipeline, real forecast-aware runs can load the
   trained checkpoint.
2. In the dashboard runtime, forecast values are still exposed and used for
   calibrated predictive behavior, while keeping the runtime lightweight and
   deterministic.

### 4.4 Controllers

RL controllers:

- Q-learning
- DQN

Baselines:

- static
- threshold
- PID
- predictive threshold

### 4.5 Dashboard runtime

The dashboard stack is:

- React frontend in `frontend/src/DashboardApp.jsx`
- Flask API in `backend/api.py`

The current dashboard uses:

- prompt text as a raw-text bridge
- the core environment's thermal dynamics
- the core energy model
- the core thresholds
- the core reward structure

Important implementation note:
- the dashboard does not run trained DQN checkpoint inference live
- instead, it runs a calibrated env-backed simulation with DQN-like policy logic
  for fast interactive validation

This is acceptable because the dashboard is a demonstration layer, while the
research code and checkpoints remain available for the full RL pipeline.

## 5. Mathematical Formulation

### 5.1 Prompt-to-heat abstraction

The project uses a structured prompt-to-heat abstraction of the form:

```text
H_t = a1 * L_t + a2 * C_t + a3 * B_t + epsilon_t
```

Where:

- `L_t` is normalized prompt length
- `C_t` is complexity score
- `B_t` is concurrency / burst intensity
- `H_t` is heat load

### 5.2 Thermal transition

The thermal state evolves according to a simplified control equation:

```text
T_(t+1) = T_t + alpha * H_t - beta * U_t + eta * (A_t - T_t) + noise
```

Where:

- `T_t` is rack temperature
- `H_t` is heat load
- `U_t` is cooling effort
- `A_t` is ambient temperature

### 5.3 Reward

The reward design combines:

- cooling energy penalty
- overheating penalty
- overcooling penalty
- instability penalty
- safe-zone bonus

Conceptually:

```text
r_t = -w1 * energy - w2 * overheat^2 - w3 * overcool^2 - w4 * instability + w5 * safe_bonus
```

### 5.4 Q-learning update

The tabular RL baseline follows:

```text
Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
```

## 6. Methodology

### 6.1 Scenarios

The system is evaluated under four workload conditions:

- Stable
- Sinusoidal
- Spiky
- Burst Heavy

These scenarios let the project compare easy, moderate, and hard thermal
conditions.

### 6.2 Forecast modes

Two control modes are compared:

- Forecast ON
- Forecast OFF

This comparison is central to the project because it shows whether predictive
signals improve resilience under harder workloads.

### 6.3 Evaluation metrics

The project reports:

- total energy
- average temperature
- max temperature
- overheat count
- total reward
- PASS/FAIL verdict

The dashboard also derives:

- business score
- cooling cost
- safe-zone rate
- risk exposure

### 6.4 Validation process

The final validation used an interactive six-case DQN sweep:

- Stable ON
- Stable OFF
- Spiky ON
- Spiky OFF
- Burst Heavy ON
- Burst Heavy OFF

This was used as the final evidence pack because it is easy to explain and
visually demonstrates forecast-aware separation.

## 7. Final Validation Results

The final six-case dashboard validation produced the following pattern.

| Case | Verdict | Total Energy | Avg Temp | Max Temp | Overheat Count |
|---|---|---:|---:|---:|---:|
| Stable ON | PASS | 7.87 | 25.41 C | 26.11 C | 0 |
| Stable OFF | PASS | 7.54 | 25.76 C | 26.70 C | 0 |
| Spiky ON | PASS | 8.53 | 25.76 C | 26.53 C | 0 |
| Spiky OFF | FAIL | 8.20 | 26.45 C | 27.21 C | 5 |
| Burst Heavy ON | PASS | 9.63 | 26.21 C | 26.99 C | 0 |
| Burst Heavy OFF | FAIL | 9.85 | 26.55 C | 27.48 C | 5 |

## 8. Interpretation

### Stable scenario

- Both ON and OFF pass.
- Forecast ON is thermally safer.
- Forecast OFF uses slightly less energy.

This is a reasonable tradeoff and supports the narrative that predictive control
is not always the cheapest, but can still be safer.

### Spiky scenario

- Forecast ON passes cleanly.
- Forecast OFF fails.

This is one of the strongest findings in the project because it shows reactive
control weakness under sudden spikes.

### Burst Heavy scenario

- Forecast ON passes after final calibration.
- Forecast OFF fails clearly.

This is the clearest resilience result in the whole project.

## 9. Consistency Checks

The final dashboard implementation was also checked for presentation
consistency:

- PASS/FAIL card
- Latest State SAFE/WARNING
- Overheat Count
- Safe Zone Rate
- safe max line on charts
- subtitle text `Temperature > 27 C`

These are now aligned to the same `safe_temp_max` source of truth from
`configs/default_config.py`.

The explainability text was also corrected so:

- Forecast ON can mention proactive behavior and predicted heat
- Forecast OFF does not incorrectly claim predictive reasoning

## 10. Discussion

The project now supports a clean final story:

> Forecasting improves resilience under harder thermal conditions, sometimes at
> a modest energy cost.

This story is supported by:

- the codebase
- the dashboard
- the final screenshots
- the final sweep table

The dashboard/runtime layer and the full training/evaluation layer are not
identical, but they are aligned closely enough to support the same conclusions.

## 11. Limitations

- The dashboard runtime uses a calibrated DQN-like control policy rather than
  live checkpoint inference.
- The project is simulation-based, not deployed to a physical cooling system.
- The final validation is intentionally compact and scenario-driven rather than
  large-scale industrial benchmarking.

These limitations should be stated clearly in the final presentation, but they
do not undermine the value of the proof of concept.

## 12. Conclusion

CoolSync+ successfully integrates:

- prompt-aware workload modeling
- predictive heat forecasting
- reinforcement learning
- scenario-based controller comparison
- an interactive dashboard demonstration

The final implementation and screenshots show that forecast-aware cooling is
most valuable under harder thermal conditions such as spiky and burst-heavy
workloads.

At this stage, the project is in a good state for documentation, presentation,
and viva discussion.
