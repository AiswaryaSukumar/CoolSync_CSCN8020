# forecasting/forecast_utils.py

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch

from forecasting.lstm_model import HeatLSTM


def load_lstm_model(
    checkpoint_path: str,
    device: torch.device,
) -> HeatLSTM:
    """
    Load a trained HeatLSTM model from checkpoint.

    Args:
        checkpoint_path: Path to the saved checkpoint file
        device: Torch device to place the model on

    Returns:
        Loaded HeatLSTM model in evaluation mode
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"LSTM checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint must contain 'config' and 'model_state_dict' keys."
        )

    model_config = checkpoint["config"]

    model = HeatLSTM(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def _prepare_history_window(
    recent_heat_history: List[float],
    sequence_length: int,
    fallback_value: float,
) -> List[float]:
    """
    Normalize recent heat history to exactly sequence_length values.

    If history is too short, pad with the latest value or fallback_value.
    If history is too long, keep only the most recent sequence_length values.
    """
    history = list(recent_heat_history)

    if len(history) < sequence_length:
        pad_value = history[-1] if history else fallback_value
        history = history + [pad_value] * (sequence_length - len(history))
    elif len(history) > sequence_length:
        history = history[-sequence_length:]

    return history


def predict_next_heat(
    recent_heat_history: List[float],
    sequence_length: int,
    forecast_model: HeatLSTM,
    device: Optional[torch.device] = None,
) -> float:
    """
    Use the trained LSTM model to predict the next heat value.

    Args:
        recent_heat_history: Recent heat values, must have exactly sequence_length items
        sequence_length: Required LSTM input window length
        forecast_model: Trained HeatLSTM model
        device: Optional torch device, defaults to CPU if omitted

    Returns:
        Predicted next-step heat value
    """
    if len(recent_heat_history) != sequence_length:
        raise ValueError(
            f"Expected recent_heat_history length {sequence_length}, "
            f"got {len(recent_heat_history)}"
        )

    if device is None:
        device = torch.device("cpu")

    input_array = np.array(
        recent_heat_history,
        dtype=np.float32,
    ).reshape(1, sequence_length, 1)

    input_tensor = torch.tensor(
        input_array,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        prediction = forecast_model(input_tensor)

    predicted_heat = float(prediction.detach().cpu().numpy().flatten()[0])

    return predicted_heat


def predict_next_heat_with_fallback(
    recent_heat_history: List[float],
    sequence_length: int,
    forecast_model: Optional[HeatLSTM] = None,
    device: Optional[torch.device] = None,
    fallback_value: float = 0.0,
) -> float:
    """
    Safe forecasting wrapper used by the environment.

    Behavior:
    - If no model is available, use the most recent observed heat
    - If history length is wrong, pad/trim to the required sequence length
    - If model inference fails, fall back to the most recent observed heat

    Args:
        recent_heat_history: Rolling heat history
        sequence_length: Required LSTM window length
        forecast_model: Optional trained model
        device: Optional torch device
        fallback_value: Fallback if no history exists

    Returns:
        Predicted or fallback next-step heat value
    """
    # No forecasting model available yet -> heuristic fallback
    if forecast_model is None:
        if recent_heat_history:
            return float(recent_heat_history[-1])
        return float(fallback_value)

    normalized_history = _prepare_history_window(
        recent_heat_history=recent_heat_history,
        sequence_length=sequence_length,
        fallback_value=fallback_value,
    )

    try:
        return predict_next_heat(
            recent_heat_history=normalized_history,
            sequence_length=sequence_length,
            forecast_model=forecast_model,
            device=device,
        )
    except Exception:
        # Safe fallback keeps the environment usable even if inference fails
        if normalized_history:
            return float(normalized_history[-1])
        return float(fallback_value)