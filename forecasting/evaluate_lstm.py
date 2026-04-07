from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from configs.default_config import CoolSyncConfig
from forecasting.lstm_dataset import HeatSequenceDataset, split_series_train_val_test
from forecasting.lstm_model import HeatLSTM
from forecasting.trace_generation import (
    DEFAULT_TRACE_SCENARIOS,
    build_heat_series_from_dataframe,
    generate_prompt_trace_dataframe,
)
from utils.seed import set_global_seed


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def load_lstm_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[HeatLSTM, Dict]:
    """
    Load a saved HeatLSTM checkpoint and return both the model and its metadata.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"LSTM checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint["config"]

    model = HeatLSTM(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, model_config


def evaluate_lstm(
    checkpoint_path: str = "results/checkpoints/lstm_heat_predictor_best.pth",
    n_plot: int = 100,
    scenarios: Sequence[str] | None = None,
    steps_per_scenario: int | None = None,
) -> Dict:
    """
    Evaluate the HeatLSTM model on prompt-driven generated test traces.
    """
    config = CoolSyncConfig()
    set_global_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, checkpoint_config = load_lstm_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    if scenarios is None:
        scenarios = checkpoint_config.get("training_scenarios", DEFAULT_TRACE_SCENARIOS)

    if steps_per_scenario is None:
        steps_per_scenario = int(checkpoint_config.get("steps_per_scenario", 600))

    trace_df = generate_prompt_trace_dataframe(
        config=config,
        scenarios=scenarios,
        steps_per_scenario=steps_per_scenario,
        base_seed=config.seed,
    )
    heat_series = build_heat_series_from_dataframe(trace_df)

    _, _, test_series = split_series_train_val_test(
        heat_series=heat_series,
        train_ratio=0.70,
        val_ratio=0.15,
    )

    test_dataset = HeatSequenceDataset(
        heat_series=test_series,
        sequence_length=config.forecast_sequence_length,
    )

    y_true_values: List[float] = []
    y_pred_values: List[float] = []

    for x_sample, y_sample in test_dataset:
        # Add batch dimension and move sample to GPU if available
        x_sample = x_sample.unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(x_sample).cpu().numpy().flatten()[0]

        target = y_sample.numpy().flatten()[0]
        y_true_values.append(float(target))
        y_pred_values.append(float(prediction))

    y_true_array = np.array(y_true_values, dtype=np.float32)
    y_pred_array = np.array(y_pred_values, dtype=np.float32)

    mae = compute_mae(y_true_array, y_pred_array)
    rmse = compute_rmse(y_true_array, y_pred_array)

    print()
    print("=== HeatLSTM Evaluation Results ===")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print()

    os.makedirs("results/plots", exist_ok=True)

    plot_path = "results/plots/lstm_actual_vs_predicted.png"

    plt.figure(figsize=(10, 5))
    plt.plot(y_true_array[:n_plot], label="Actual Heat")
    plt.plot(y_pred_array[:n_plot], label="Predicted Heat")
    plt.xlabel("Time Step")
    plt.ylabel("Heat")
    plt.title("HeatLSTM Evaluation — Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved : {plot_path}")

    # Save evaluation summary for reproducibility and report evidence
    os.makedirs("results/summaries", exist_ok=True)

    metrics_summary = {
        "mae": mae,
        "rmse": rmse,
        "num_test_samples": len(test_dataset),
        "plot_path": plot_path,
        "evaluation_scenarios": list(scenarios),
        "steps_per_scenario": int(steps_per_scenario),
        "checkpoint_path": checkpoint_path,
        "device": str(device),
    }

    with open("results/summaries/lstm_evaluation.json", "w", encoding="utf-8") as file:
        json.dump(metrics_summary, file, indent=2)

    return metrics_summary


if __name__ == "__main__":
    evaluate_lstm()