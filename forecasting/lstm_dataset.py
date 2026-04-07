# forecasting/lstm_dataset.py

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def build_lstm_sequences(
    heat_series: Sequence[float],
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build one-step forecasting windows from a 1D heat series.

    Example:
        X = [heat_t-9, ..., heat_t]
        y = [heat_t+1]

    Returns:
        x_array of shape (num_samples, sequence_length, 1)
        y_array of shape (num_samples, 1)
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than 0.")

    if len(heat_series) <= sequence_length:
        raise ValueError(
            "Heat series length must be greater than sequence_length. "
            f"Got len(heat_series)={len(heat_series)} and sequence_length={sequence_length}."
        )

    x_sequences = []
    y_targets = []

    for start_idx in range(len(heat_series) - sequence_length):
        end_idx = start_idx + sequence_length

        # Input window: recent heat history
        x_window = heat_series[start_idx:end_idx]

        # Target: next-step heat
        y_target = heat_series[end_idx]

        x_sequences.append(x_window)
        y_targets.append(y_target)

    x_array = np.array(x_sequences, dtype=np.float32).reshape(-1, sequence_length, 1)
    y_array = np.array(y_targets, dtype=np.float32).reshape(-1, 1)

    return x_array, y_array


class HeatSequenceDataset(Dataset):
    """
    PyTorch dataset for one-step heat forecasting.

    Each sample:
        x -> shape (sequence_length, 1)
        y -> shape (1,)
    """

    def __init__(
        self,
        heat_series: Sequence[float],
        sequence_length: int,
    ) -> None:
        self.x_array, self.y_array = build_lstm_sequences(
            heat_series=heat_series,
            sequence_length=sequence_length,
        )

    def __len__(self) -> int:
        return len(self.x_array)

    def __getitem__(self, idx: int):
        x_tensor = torch.tensor(self.x_array[idx], dtype=torch.float32)
        y_tensor = torch.tensor(self.y_array[idx], dtype=torch.float32)
        return x_tensor, y_tensor


def split_series_train_val_test(
    heat_series: Sequence[float],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Split a 1D heat series into train / validation / test segments.

    The remaining portion after train and validation becomes test.
    """
    total_length = len(heat_series)

    if total_length < 20:
        raise ValueError("Heat series is too short for a meaningful train/val/test split.")

    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1.")

    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0 and 1.")

    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0.")

    train_end = int(total_length * train_ratio)
    val_end = int(total_length * (train_ratio + val_ratio))

    train_series = list(heat_series[:train_end])
    val_series = list(heat_series[train_end:val_end])
    test_series = list(heat_series[val_end:])

    return train_series, val_series, test_series