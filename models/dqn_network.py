# models/dqn_network.py

from __future__ import annotations

import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """
    Simple fully connected Deep Q-Network for CoolSync+.

    This network takes a normalized state vector and predicts one Q-value
    per action.

    Notes:
    - Works for both forecast-off and forecast-on states
    - Uses CUDA automatically when the caller moves the model to GPU
    - Small and stable architecture for tabular-to-deep transition
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 128,
    ) -> None:
        """
        Initialize the DQN network.

        Args:
            input_dim: Number of state features.
            output_dim: Number of actions.
            hidden_dim_1: Width of first hidden layer.
            hidden_dim_2: Width of second hidden layer.
        """
        super().__init__()

        # Define a compact MLP for Q-value approximation.
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Q-network.

        Args:
            x: Input state tensor of shape (batch_size, input_dim)
               or (input_dim,) for a single state.

        Returns:
            Tensor of Q-values with shape:
            - (batch_size, output_dim), or
            - (output_dim,) for a single state
        """
        return self.network(x)