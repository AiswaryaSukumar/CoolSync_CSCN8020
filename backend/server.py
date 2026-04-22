"""
CoolSync FastAPI backend — serves DQN+LSTM simulation to GreenInference UI.

Start:
    pip install fastapi uvicorn torch numpy
    cd backend
    uvicorn server:app --reload --port 8000

The Vite dev server proxies /api/* → http://localhost:8000 (see vite.config.js).
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from typing import List, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Resolve model paths relative to this file
BACKEND_DIR  = Path(__file__).parent
MODELS_DIR   = BACKEND_DIR.parent / "CoolSync_final" / "models"

# ─── Load PyTorch models (lazy import so server starts fast if torch missing) ─
_models_loaded = False
_lstm_model    = None
_dqn_model     = None
_device        = None
_joint_actions = None


def _load_models():
    global _models_loaded, _lstm_model, _dqn_model, _device, _joint_actions
    if _models_loaded:
        return True
    try:
        import torch
        import torch.nn as nn

        _device = torch.device("cpu")

        class LSTMPredictor(nn.Module):
            def __init__(self, input_size=3, hidden=48):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden, num_layers=2,
                                    batch_first=True, dropout=0.1)
                self.head = nn.Linear(hidden, 1)
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])

        class DQNNet(nn.Module):
            def __init__(self, obs_dim=13, hidden=64, n_actions=10):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(obs_dim, hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden),  nn.ReLU(),
                    nn.Linear(hidden, n_actions),
                )
            def forward(self, x):
                return self.net(x)

        lstm_path = MODELS_DIR / "lstm.pt"
        dqn_path  = MODELS_DIR / "dqn.pt"

        if not lstm_path.exists() or not dqn_path.exists():
            print(f"[server] Model files not found in {MODELS_DIR}", file=sys.stderr)
            return False

        lstm_ckpt = torch.load(lstm_path, map_location=_device, weights_only=False)
        _lstm_model = LSTMPredictor()
        _lstm_model.load_state_dict(lstm_ckpt["state_dict"])
        _lstm_model.eval()

        dqn_ckpt = torch.load(dqn_path, map_location=_device, weights_only=False)
        _dqn_model = DQNNet()
        _dqn_model.load_state_dict(dqn_ckpt["online_state_dict"])
        _dqn_model.eval()

        from physics import JOINT_ACTIONS
        _joint_actions = JOINT_ACTIONS

        _models_loaded = True
        print(f"[server] Models loaded from {MODELS_DIR}", file=sys.stderr)
        return True
    except Exception as exc:
        print(f"[server] Model load failed: {exc}", file=sys.stderr)
        return False


# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="CoolSync Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://127.0.0.1:5174"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    _load_models()


# ─── Request / Response schemas ───────────────────────────────────────────────
Strategy = Literal["pid", "pid_conservative", "dqn", "coordinated"]


class SimRequest(BaseModel):
    heat_trace:  List[float]
    strategies:  List[Strategy] = ["pid", "pid_conservative", "dqn"]


class RunResult(BaseModel):
    points:       List[dict]
    energy_wh:    float
    peak_T:       float
    breach_steps: int
    fan_var:      float


class BatchResponse(BaseModel):
    pid:              RunResult | None = None
    pid_conservative: RunResult | None = None
    dqn:              RunResult | None = None
    coordinated:      RunResult | None = None


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status":       "ok",
        "models_loaded": _models_loaded,
        "models_dir":   str(MODELS_DIR),
    }


@app.post("/api/simulate/batch", response_model=BatchResponse)
def simulate_batch(req: SimRequest):
    from physics import (
        run_pid, run_pid_conservative, run_coordinated, run_dqn, summarise,
    )

    heat = req.heat_trace
    if len(heat) == 0:
        raise HTTPException(status_code=422, detail="heat_trace must not be empty")

    result = {}

    for strat in req.strategies:
        if strat == "pid":
            rows  = run_pid(heat)
        elif strat == "pid_conservative":
            rows  = run_pid_conservative(heat)
        elif strat == "coordinated":
            rows  = run_coordinated(heat)
        elif strat == "dqn":
            if not _models_loaded or not _load_models():
                # Fallback to coordinated mode if models unavailable
                rows = run_coordinated(heat)
            else:
                rows = run_dqn(heat, _dqn_model, _lstm_model, _joint_actions, _device)
        else:
            continue

        stats = summarise(rows)
        result[strat] = RunResult(points=rows, **stats)

    return BatchResponse(**result)
