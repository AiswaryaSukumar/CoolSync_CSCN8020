# training/state_preprocessor.py

from __future__ import annotations

from typing import Tuple
import numpy as np

from configs.default_config import CoolSyncConfig


class StatePreprocessor:
    """
    Utility class for:
    1. Normalizing continuous state values for DQN
    2. Discretizing continuous state values for Q-learning

    State contract:
        [
            current_temperature,
            current_workload,
            current_cooling_level,
            ambient_temperature,
            previous_action,
            predicted_heat_next_step,
        ]

    If use_forecast=False, the forecast dimension is ignored.
    """

    def __init__(self, config: CoolSyncConfig, use_forecast: bool = True) -> None:
        self.config = config
        self.use_forecast = use_forecast

        # Bin counts for tabular Q-learning
        self.temp_bins = 12
        self.workload_bins = 10
        self.cooling_bins = config.max_cooling_level - config.min_cooling_level + 1
        self.ambient_bins = 8
        self.prev_action_bins = 3
        self.predicted_heat_bins = 10

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize the environment state into approximately [0, 1].
        """
        current_temperature = state[0]
        current_workload = state[1]
        current_cooling_level = state[2]
        ambient_temperature = state[3]
        previous_action = state[4]
        predicted_heat_next_step = state[5] if len(state) > 5 else 0.0

        norm_temperature = (
            (current_temperature - self.config.temp_min_for_norm)
            / (self.config.temp_max_for_norm - self.config.temp_min_for_norm)
        )

        norm_workload = (
            (current_workload - self.config.workload_min_for_norm)
            / (self.config.workload_max_for_norm - self.config.workload_min_for_norm)
        )

        norm_cooling = (
            (current_cooling_level - self.config.min_cooling_level)
            / (self.config.max_cooling_level - self.config.min_cooling_level)
        )

        norm_ambient = (
            (ambient_temperature - self.config.temp_min_for_norm)
            / (self.config.temp_max_for_norm - self.config.temp_min_for_norm)
        )

        norm_prev_action = (
            (previous_action - self.config.action_min_for_norm)
            / (self.config.action_max_for_norm - self.config.action_min_for_norm)
        )

        norm_predicted_heat = (
            (predicted_heat_next_step - self.config.heat_min_for_norm)
            / (self.config.heat_max_for_norm - self.config.heat_min_for_norm)
        )

        if self.use_forecast:
            normalized = np.array(
                [
                    np.clip(norm_temperature, 0.0, 1.0),
                    np.clip(norm_workload, 0.0, 1.0),
                    np.clip(norm_cooling, 0.0, 1.0),
                    np.clip(norm_ambient, 0.0, 1.0),
                    np.clip(norm_prev_action, 0.0, 1.0),
                    np.clip(norm_predicted_heat, 0.0, 1.0),
                ],
                dtype=np.float32,
            )
        else:
            normalized = np.array(
                [
                    np.clip(norm_temperature, 0.0, 1.0),
                    np.clip(norm_workload, 0.0, 1.0),
                    np.clip(norm_cooling, 0.0, 1.0),
                    np.clip(norm_ambient, 0.0, 1.0),
                    np.clip(norm_prev_action, 0.0, 1.0),
                ],
                dtype=np.float32,
            )

        return normalized

    def discretize_state(self, state: np.ndarray) -> Tuple[int, ...]:
        """
        Convert continuous state into a tuple of discrete bin indices for Q-learning.
        """
        current_temperature = state[0]
        current_workload = state[1]
        current_cooling_level = state[2]
        ambient_temperature = state[3]
        previous_action = state[4]
        predicted_heat_next_step = state[5] if len(state) > 5 else 0.0

        temp_idx = self._digitize(
            value=current_temperature,
            low=self.config.temp_min_for_norm,
            high=self.config.temp_max_for_norm,
            bins=self.temp_bins,
        )

        workload_idx = self._digitize(
            value=current_workload,
            low=self.config.workload_min_for_norm,
            high=self.config.workload_max_for_norm,
            bins=self.workload_bins,
        )

        cooling_idx = int(
            np.clip(
                current_cooling_level,
                self.config.min_cooling_level,
                self.config.max_cooling_level,
            )
        )

        ambient_idx = self._digitize(
            value=ambient_temperature,
            low=self.config.temp_min_for_norm,
            high=self.config.temp_max_for_norm,
            bins=self.ambient_bins,
        )

        prev_action_idx = int(np.clip(previous_action, 0, 2))

        predicted_heat_idx = self._digitize(
            value=predicted_heat_next_step,
            low=self.config.heat_min_for_norm,
            high=self.config.heat_max_for_norm,
            bins=self.predicted_heat_bins,
        )

        if self.use_forecast:
            return (
                temp_idx,
                workload_idx,
                cooling_idx,
                ambient_idx,
                prev_action_idx,
                predicted_heat_idx,
            )

        return (
            temp_idx,
            workload_idx,
            cooling_idx,
            ambient_idx,
            prev_action_idx,
        )

    @staticmethod
    def _digitize(value: float, low: float, high: float, bins: int) -> int:
        """
        Map a continuous value to an integer bin.
        """
        edges = np.linspace(low, high, bins + 1)
        idx = np.digitize(value, edges[1:-1], right=False)
        return int(np.clip(idx, 0, bins - 1))