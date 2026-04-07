# tests/smoke_test_lstm.py

from __future__ import annotations

import os
import sys

# Add project root to Python path so package-style imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import os as _os

from forecasting.train_lstm import train_lstm
from forecasting.evaluate_lstm import evaluate_lstm


def test_lstm_smoke() -> None:
    """
    Smoke test for LSTM training and evaluation.

    Verifies:
    - training runs for a small number of epochs
    - checkpoint is created
    - evaluation runs on the saved checkpoint
    """
    result = train_lstm(
        epochs=2,
        learning_rate=1e-3,
        batch_size=16,
    )

    checkpoint_path = result["checkpoint_path"]

    assert _os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

    eval_result = evaluate_lstm(checkpoint_path=checkpoint_path)

    assert "mae" in eval_result, "MAE missing from evaluation result"
    assert "rmse" in eval_result, "RMSE missing from evaluation result"
    assert eval_result["num_test_samples"] > 0, "No test samples evaluated"

    print("[PASS] smoke_test_lstm")

if __name__ == "__main__":
    test_lstm_smoke()