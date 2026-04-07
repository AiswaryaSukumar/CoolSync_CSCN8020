# CoolSync+: Prompt-Aware Predictive Cooling for AI Data Centers

## Overview

CoolSync+ is a simulation-driven cooling optimization project for AI data centers.
It combines:

- prompt-aware workload and heat generation
- one-step heat forecasting with LSTM
- reinforcement learning controllers
- baseline rule-based controllers
- a dashboard demo for scenario validation

The project goal is to show that forecast-aware control can improve thermal
resilience under harder workloads while still managing energy use responsibly.

## Current Project Truth

The repository contains two closely related layers:

1. Research and training layer
- prompt-driven trace generation
- LSTM training and evaluation
- Q-learning and DQN training
- multi-controller comparisons across scenarios

2. Dashboard runtime layer
- React frontend in `frontend/`
- Flask API in `backend/api.py`
- env-backed 16-step simulation episodes for interactive demos

Important:
- The dashboard now uses the core environment's temperature dynamics, reward,
  thresholds, and energy model.
- The dashboard does not load trained DQN checkpoints for live inference.
  Instead, it uses a calibrated DQN-like control policy inside the backend so
  the interactive UI remains deterministic, explainable, and fast.
- The research pipeline still supports real trained LSTM and RL checkpoints.

## End-to-End Architecture

```text
Prompt category / raw prompt bridge
-> prompt features
-> workload and heat generation
-> thermal environment
-> one-step forecast
-> controller action
-> cooling, energy, reward, and safety metrics
```

For the main research pipeline:

```text
Prompt trace generation
-> LSTM training data
-> LSTM forecast model
-> CoolSyncEnv
-> Q-Learning / DQN / baselines
-> logs, summaries, plots, checkpoints
```

For the dashboard:

```text
DashboardApp.jsx
-> POST /api/run-simulation
-> backend/api.py
-> env-backed 16-step episode
-> prompt_features + summary + timeseries
```

## Key Features

### Prompt-aware simulation

- Prompt types: `simple`, `reasoning`, `code`, `long_context`, `burst`
- Structured prompt features:
  - `prompt_type`
  - `prompt_length`
  - `complexity_score`
  - `concurrency_level`
- Prompt-to-heat conversion in `prompt_model/`

### Scenario framework

Supported scenarios:

- `stable`
- `sinusoidal`
- `spiky`
- `burst_heavy`

These scenarios shape workload variation, ambient behavior, and heat pressure.

### Thermal environment

The core environment is implemented in `envs/coolsync_env.py`.

It includes:

- discrete cooling actions: decrease / maintain / increase
- nonlinear cooling energy model
- ambient temperature coupling
- safe and critical temperature thresholds
- reward terms for energy, overheating, overcooling, instability, and safe operation

### Forecasting

- one-step LSTM forecasting
- training and evaluation under `forecasting/`
- real checkpoint support through:

```text
results/checkpoints/lstm_heat_predictor_best.pth
```

### Controllers

RL controllers:

- Q-learning
- DQN

Baselines:

- static controller
- threshold controller
- PID controller
- predictive threshold controller

### Dashboard validation

The dashboard provides:

- interactive prompt simulation
- forecast on/off comparison
- PASS/FAIL business verdict
- temperature, heat, energy, reward, and cumulative reward charts
- controller benchmark comparison

## Final Validation Story

The final dashboard sweep is designed to support this interpretation:

- Stable ON: PASS, thermally safest stable run
- Stable OFF: PASS, slightly cheaper energy but less proactive
- Spiky ON: PASS, forecast keeps spikes controlled
- Spiky OFF: FAIL or borderline FAIL, showing reactive weakness
- Burst Heavy ON: PASS or borderline PASS, recovered by proactive cooling
- Burst Heavy OFF: FAIL, worst-case reactive behavior

This supports the project's main conclusion:

> Forecasting improves resilience under harder thermal conditions, sometimes at
> a modest energy cost.

## Repository Structure

```text
CoolSync_CSCN8020/
|-- agents/                 RL agents
|-- backend/                Flask dashboard API
|-- baselines/              Rule-based controllers
|-- configs/                Project configuration
|-- data/                   Generated and source data
|-- envs/                   Thermal environment
|-- forecasting/            LSTM forecasting pipeline
|-- frontend/               React dashboard
|-- models/                 Neural network models
|-- prompt_model/           Prompt-aware simulation layer
|-- report/                 Report documents
|-- results/                Logs, plots, summaries, checkpoints
|-- scenarios/              Scenario definitions
|-- tests/                  Smoke tests
|-- training/               RL utilities
|-- utils/                  Shared helpers
|-- compare_controllers.py
|-- main.py
|-- run_experiments.py
|-- train_dqn.py
|-- train_q_learning.py
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run

### Full pipeline

```bash
python main.py --mode full
```

### Step-by-step

Generate prompt-driven traces:

```bash
python main.py --mode generate_data
```

Train LSTM:

```bash
python main.py --mode train_lstm
```

Evaluate LSTM:

```bash
python main.py --mode evaluate_lstm
```

Train Q-learning:

```bash
python main.py --mode train_q_learning --scenario stable
python main.py --mode train_q_learning --scenario stable --use_forecast
```

Train DQN:

```bash
python main.py --mode train_dqn --scenario stable
python main.py --mode train_dqn --scenario stable --use_forecast
```

Compare controllers:

```bash
python main.py --mode compare --scenario stable --episodes 5
```

Run experiment sweep:

```bash
python main.py --mode experiments --episodes 5
```

Run the dashboard backend:

```bash
python backend/api.py
```

Run the dashboard frontend:

```bash
cd frontend
npm install
npm run dev
```

## Core Equations

Prompt-to-heat intuition:

```text
heat_t = a1 * length_t + a2 * complexity_t + a3 * concurrency_t + noise
```

Thermal transition:

```text
T(t+1) = T(t) + alpha * heat - beta * cooling + ambient_coupling + noise
```

Reward structure:

```text
reward = -energy - overheat_penalty - overcool_penalty - instability_penalty + safe_bonus
```

Q-learning update:

```text
Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
```

## Outputs

Main output folders:

- `results/checkpoints/`
- `results/logs/`
- `results/summaries/`
- `results/comparisons/`
- `results/plots/`

Important artifacts:

- generated traces:
  - `data/generated/prompt_driven_heat_traces.csv`
- best LSTM checkpoint:
  - `results/checkpoints/lstm_heat_predictor_best.pth`
- controller summaries:
  - `results/summaries/`
- benchmark plots:
  - `results/comparisons/`

## Validation and Reproducibility

The repository includes smoke tests for major components:

```bash
pytest -q
```

The project also uses:

- reproducibility seed control in `utils/seed.py`
- structured logs and summaries under `results/`
- explicit scenario definitions

## Final Takeaway

CoolSync+ demonstrates a coherent prompt-aware predictive cooling workflow:

- prompt patterns drive workload and heat
- forecasting provides future-aware thermal signals
- RL and baseline controllers can be compared across scenarios
- proactive control becomes most valuable in harder thermal conditions

The code, dashboard, and final validation screenshots are now aligned closely
enough for report writing and presentation.
