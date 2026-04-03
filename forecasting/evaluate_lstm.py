# forecasting/evaluate_lstm.py

import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from forecasting.data_loader import LSTMDataLoader
from forecasting.lstm_model  import ThermalLSTM
from configs.default_config  import CoolSyncConfig


def evaluate_lstm(
    csv_path:        str = 'data/merged_lstm_core.csv',
    checkpoint_path: str = 'results/checkpoints/lstm_best.pth',
    lookback:        int = 8,
    n_plot:          int = 200,
):
    """
    Evaluates the trained LSTM model.

    Loads the best saved checkpoint and runs predictions
    on the full dataset. Calculates MAE, RMSE, and R2.
    Saves a plot of predicted vs actual thermal load.
    """

    config = CoolSyncConfig()

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Load trained model
    model = ThermalLSTM(
        input_size  = config.lstm_input_size,
        hidden_size = config.lstm_hidden,
        num_layers  = config.lstm_layers,
        dropout     = config.lstm_dropout,
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )
    model.to(device)
    model.eval()

    print(f"Model loaded from : {checkpoint_path}")

    # Load full dataset
    loader       = LSTMDataLoader(csv_path, lookback=lookback)
    X_all, y_all = loader.get_full_sequence()

    # Run predictions
    X_tensor = torch.FloatTensor(X_all).to(device)

    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()

    # Calculate metrics
    mae  = float(np.mean(np.abs(preds - y_all)))
    rmse = float(np.sqrt(np.mean((preds - y_all) ** 2)))

    ss_res = np.sum((y_all - preds) ** 2)
    ss_tot = np.sum((y_all - np.mean(y_all)) ** 2)
    r2     = float(1 - (ss_res / ss_tot))

    print()
    print("=== LSTM Evaluation Results ===")
    print(f"MAE  : {mae:.4f}  (lower is better)")
    print(f"RMSE : {rmse:.4f}  (lower is better)")
    print(f"R2   : {r2:.4f}  (1.0 = perfect)")
    print()

    if r2 > 0.8:
        print("Result : Excellent - model explains >80% of variance")
    elif r2 > 0.6:
        print("Result : Good - model explains >60% of variance")
    elif r2 > 0.4:
        print("Result : Moderate - model has some predictive power")
    else:
        print("Result : Weak - model needs improvement")

    # Plot predicted vs actual
    os.makedirs('results/plots', exist_ok=True)

    plt.figure(figsize=(14, 5))
    plt.plot(
        y_all[:n_plot],
        label='Actual TLHC',
        color='blue',
        linewidth=1.5
    )
    plt.plot(
        preds[:n_plot],
        label='Predicted TLHC',
        color='orange',
        linewidth=1.5,
        linestyle='--'
    )
    plt.xlabel('Timestep')
    plt.ylabel('Thermal Load (TLHC normalized)')
    plt.title(
        f'LSTM Evaluation — Predicted vs Actual\n'
        f'MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}'
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/plots/lstm_evaluation.png')
    plt.close()

    print(f"Plot saved : results/plots/lstm_evaluation.png")

    return {
        'mae':  mae,
        'rmse': rmse,
        'r2':   r2,
    }