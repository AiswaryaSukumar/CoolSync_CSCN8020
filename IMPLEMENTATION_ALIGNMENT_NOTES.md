# Implementation Alignment Notes

These notes capture the current project truth so the code, dashboard, report,
and presentation stay synchronized.

## 1. Source of Truth

Use the current repository files as the source of truth.

Do not overwrite the current implementation with older skeleton snippets or
earlier intermediate drafts. Some older versions:

- used weaker orchestration
- treated forecasting as placeholder logic
- described the project at a higher level than the implemented code
- did not reflect the final dashboard/runtime integration work

## 2. Project Framing

The final framing is:

```text
Prompt-aware workload generation + forecasting + RL cooling control
```

This should remain visible in all written documentation.

The project should not be described as only:

- a plain cooling controller
- only an LSTM forecasting project
- only a dashboard visualization

It is the combination that matters.

## 3. Research Layer vs Dashboard Layer

The repository contains two aligned but distinct layers.

### Research and training layer

This includes:

- prompt-driven data generation
- LSTM training and evaluation
- Q-learning and DQN training
- controller comparisons
- scenario-based experiment sweeps

Relevant files:

- `main.py`
- `train_q_learning.py`
- `train_dqn.py`
- `compare_controllers.py`
- `forecasting/`
- `agents/`
- `envs/coolsync_env.py`

### Dashboard runtime layer

This includes:

- `frontend/src/DashboardApp.jsx`
- `backend/api.py`

The dashboard now uses:

- the environment's thresholds
- the environment's reward calculation
- the environment's energy model
- the environment's temperature dynamics

But the dashboard does not perform live trained DQN checkpoint inference.
Instead, it uses an env-backed calibrated DQN-like controller for interactive
runtime behavior.

This distinction should be documented honestly.

## 4. Forecasting Alignment

Forecast-aware research runs should keep using the trained LSTM checkpoint:

```text
results/checkpoints/lstm_heat_predictor_best.pth
```

This is still the intended checkpoint path for:

- `train_q_learning.py`
- `train_dqn.py`
- `main.py`
- `compare_controllers.py`

## 5. Dashboard API Contract

The current dashboard simulation route is:

```text
POST /api/run-simulation
```

The response contract that the frontend expects is:

- `prompt_features`
- `summary`
- `timeseries`

This contract should be preserved unless there is a deliberate frontend update.

The backend now also includes `summary.safe_temp_max` so the frontend can stay
aligned with the environment threshold.

## 6. Final Threshold Source of Truth

Use `configs/default_config.py` as the threshold source of truth.

Important values:

- `safe_temp_min`
- `safe_temp_max`
- `critical_temp`

The frontend should not introduce competing hardcoded threshold logic.

## 7. Final Validation Story

The final dashboard sweep now supports this pattern:

- Stable ON: PASS
- Stable OFF: PASS
- Spiky ON: PASS
- Spiky OFF: FAIL
- Burst Heavy ON: PASS
- Burst Heavy OFF: FAIL

This is the project's strongest presentation story and should be preserved in:

- `README.md`
- `report/report.md`
- `report/methodology.md`
- slides or viva notes

## 8. Explainability Alignment

The "Why this action?" text in the frontend was corrected so that:

- Forecast ON can reference proactive control and predicted heat
- Forecast OFF uses reactive wording and does not pretend to be predictive

If this text changes later, keep it mode-honest.

## 9. What Not to Claim

Do not claim that:

- the dashboard directly runs the trained DQN checkpoint online
- the system is validated on a live physical data center
- the final validation is a production deployment study

The accurate claim is:

> This is a simulation-based prompt-aware predictive cooling prototype with
> trained forecasting and RL support, plus an env-backed interactive dashboard.

## 10. Documentation Rule

All documentation should now reflect:

- prompt-aware simulation
- one-step forecasting
- RL plus baseline controllers
- env-backed dashboard runtime
- final six-case validation behavior

At this point, logic should stay stable and the focus should shift to
documentation and presentation.
