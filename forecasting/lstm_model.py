# forecasting/lstm_model.py

from __future__ import annotations

import torch
import torch.nn as nn


class HeatLSTM(nn.Module):
    """
    LSTM model for one-step heat forecasting.

    Expected input shape:
        (batch_size, sequence_length, input_dim)

    Expected output shape:
        (batch_size, 1)

    Default design:
    - input_dim = 1 because the dataset uses a 1D heat history sequence
    - output predicts next-step heat
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # LSTM backbone for sequence modeling
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Optional dropout before final regression
        self.dropout = nn.Dropout(dropout)

        # Final regression layer predicting one next-step heat value
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch_size, sequence_length, input_dim)

        Returns:
            Tensor of shape (batch_size, 1)
        """
        # LSTM output shape: (batch_size, sequence_length, hidden_dim)
        lstm_output, _ = self.lstm(x)

        # Take representation from final timestep
        final_timestep_output = lstm_output[:, -1, :]

        # Apply dropout before regression head
        final_timestep_output = self.dropout(final_timestep_output)

        # Predict next-step heat
        prediction = self.output_layer(final_timestep_output)

        return prediction

    def count_parameters(self) -> int:
        """
        Return total number of trainable parameters.
        """
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )