from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from configs.default_config import CoolSyncConfig
from forecasting.lstm_dataset import HeatSequenceDataset, split_series_train_val_test
from forecasting.lstm_model import HeatLSTM
from forecasting.trace_generation import (
    DEFAULT_TRACE_SCENARIOS,
    build_heat_series_from_dataframe,
    generate_and_save_trace_dataset,
    generate_prompt_trace_dataframe,
)
from utils.seed import set_global_seed


def build_prompt_driven_heat_series(
    config: CoolSyncConfig,
    scenarios: Sequence[str] | None = None,
    steps_per_scenario: int = 600,
    save_csv: bool = True,
) -> List[float]:
    """
    Build the LSTM training signal directly from the project's prompt-aware
    simulation layer instead of using any dummy synthetic series.
    """
    if scenarios is None:
        scenarios = DEFAULT_TRACE_SCENARIOS

    if save_csv:
        generate_and_save_trace_dataset(
            config=config,
            scenarios=scenarios,
            steps_per_scenario=steps_per_scenario,
            output_path="data/generated/prompt_driven_heat_traces.csv",
        )

    trace_df = generate_prompt_trace_dataframe(
        config=config,
        scenarios=scenarios,
        steps_per_scenario=steps_per_scenario,
        base_seed=config.seed,
    )
    return build_heat_series_from_dataframe(trace_df)


def build_dataloaders(
    config: CoolSyncConfig,
    batch_size: int = 32,
    scenarios: Sequence[str] | None = None,
    steps_per_scenario: int = 600,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / validation / test dataloaders from prompt-driven generated traces.
    """
    heat_series = build_prompt_driven_heat_series(
        config=config,
        scenarios=scenarios,
        steps_per_scenario=steps_per_scenario,
        save_csv=True,
    )

    train_series, val_series, test_series = split_series_train_val_test(
        heat_series=heat_series,
        train_ratio=0.70,
        val_ratio=0.15,
    )

    train_dataset = HeatSequenceDataset(
        heat_series=train_series,
        sequence_length=config.forecast_sequence_length,
    )
    val_dataset = HeatSequenceDataset(
        heat_series=val_series,
        sequence_length=config.forecast_sequence_length,
    )
    test_dataset = HeatSequenceDataset(
        heat_series=test_series,
        sequence_length=config.forecast_sequence_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def run_one_epoch(
    model: HeatLSTM,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
) -> float:
    """
    Run one epoch of either training or validation.
    """
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    batch_losses: List[float] = []

    for x_batch, y_batch in dataloader:
        # Move tensors to GPU if CUDA is available
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            predictions = model(x_batch)
            loss = loss_fn(predictions, y_batch)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_losses.append(float(loss.item()))

    return float(sum(batch_losses) / len(batch_losses)) if batch_losses else 0.0


def train_lstm(
    epochs: int = 20,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    save_path: str = "results/checkpoints/lstm_heat_predictor_best.pth",
    scenarios: Sequence[str] | None = None,
    steps_per_scenario: int = 600,
) -> Dict:
    """
    Train the HeatLSTM model on prompt-driven generated heat traces and save
    the best checkpoint based on validation loss.
    """
    config = CoolSyncConfig()
    set_global_seed(config.seed)

    if scenarios is None:
        scenarios = DEFAULT_TRACE_SCENARIOS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"LSTM training device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(
        config=config,
        batch_size=batch_size,
        scenarios=scenarios,
        steps_per_scenario=steps_per_scenario,
    )

    model = HeatLSTM(
        input_dim=1,
        hidden_dim=32,
        num_layers=1,
        dropout=0.0,
    ).to(device)

    print(f"Trainable parameters: {model.count_parameters():,}")

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val_loss = float("inf")

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = run_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        val_loss = run_one_epoch(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            optimizer=None,
            device=device,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": {
                    "input_dim": 1,
                    "hidden_dim": 32,
                    "num_layers": 1,
                    "dropout": 0.0,
                    "sequence_length": config.forecast_sequence_length,
                    "training_scenarios": list(scenarios),
                    "steps_per_scenario": int(steps_per_scenario),
                },
            }
            torch.save(checkpoint, save_path)

        print(
            f"[LSTM] Epoch {epoch}/{epochs} | "
            f"TrainLoss={train_loss:.6f} | "
            f"ValLoss={val_loss:.6f} | "
            f"BestVal={best_val_loss:.6f}"
        )

    # Save training summary for reproducibility and report evidence
    os.makedirs("results/summaries", exist_ok=True)

    training_summary = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "checkpoint_path": save_path,
        "device": str(device),
        "num_test_batches": len(test_loader),
        "training_scenarios": list(scenarios),
        "steps_per_scenario": int(steps_per_scenario),
        "trace_dataset_path": "data/generated/prompt_driven_heat_traces.csv",
    }

    with open("results/summaries/lstm_training_summary.json", "w", encoding="utf-8") as file:
        json.dump(training_summary, file, indent=2)

    return training_summary


if __name__ == "__main__":
    train_lstm()