# forecasting/predict.py

import numpy as np
import torch

from forecasting.lstm_model import ThermalLSTM
from configs.default_config import CoolSyncConfig


class HeatPredictor:
    """
    Wrapper around the trained ThermalLSTM.

    Used by the RL environment at every timestep
    to get the predicted next thermal load.

    The environment does not need to know anything
    about the LSTM internals. It just calls predict()
    and receives one float value back.

    Usage:
        predictor = HeatPredictor('results/checkpoints/lstm_best.pth')
        predicted_heat = predictor.predict(history_array)
    """

    def __init__(
        self,
        checkpoint_path: str,
    ):
        config = CoolSyncConfig()

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Build model with same architecture used in training
        self.model = ThermalLSTM(
            input_size  = config.lstm_input_size,
            hidden_size = config.lstm_hidden,
            num_layers  = config.lstm_layers,
            dropout     = config.lstm_dropout,
        )

        # Load saved weights
        self.model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=self.device
            )
        )
        self.model.to(self.device)
        self.model.eval()

        self.checkpoint_path = checkpoint_path
        print(f"HeatPredictor loaded from : {checkpoint_path}")

    def predict(self, history: np.ndarray) -> float:
        """
        Predict next thermal load from recent history.

        Args:
            history : numpy array of shape (lookback, features)
                      e.g. (8, 10) for 8 timesteps x 10 features
                      Must be normalized the same way as training data

        Returns:
            predicted TLHC value as a float
        """
        if len(history) < 1:
            return 0.0

        # Add batch dimension (1, lookback, features)
        x = torch.FloatTensor(history).unsqueeze(0)
        x = x.to(self.device)

        with torch.no_grad():
            prediction = self.model(x)

        return float(prediction.item())

    def predict_is_spike(
        self,
        history:   np.ndarray,
        threshold: float = 1.0
    ) -> bool:
        """
        Returns True if predicted thermal load
        is above the given threshold.

        Used by the environment to flag
        incoming heat spikes so the RL agent
        can act proactively.

        threshold = 1.0 means one standard deviation
        above the mean thermal load
        """
        return self.predict(history) > threshold

    def __repr__(self) -> str:
        return (
            f"HeatPredictor("
            f"checkpoint='{self.checkpoint_path}', "
            f"device='{self.device}')"
        )