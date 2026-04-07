# compare_results.py

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(path: str) -> None:
    """
    Create a directory if it does not already exist.
    """
    if path:
        os.makedirs(path, exist_ok=True)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load a JSON file and return it as a dictionary.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Parsed JSON dictionary.
    """
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)


def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Safely fetch a key from a dictionary.

    Args:
        data: Input dictionary.
        key: Key to fetch.
        default: Default value if key is missing.

    Returns:
        Value if found, otherwise default.
    """
    return data[key] if key in data else default


def build_result_row(
    model_name: str,
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert one evaluation summary into a flat comparison row.

    Args:
        model_name: Friendly name for the model/controller.
        summary: Evaluation summary dictionary.

    Returns:
        Flat row dictionary for DataFrame/CSV use.
    """
    return {
        "model": model_name,
        "use_forecast": safe_get(summary, "use_forecast"),
        "episodes_evaluated": safe_get(summary, "episodes_evaluated"),
        "mean_reward": safe_get(summary, "mean_reward"),
        "std_reward": safe_get(summary, "std_reward"),
        "min_reward": safe_get(summary, "min_reward"),
        "max_reward": safe_get(summary, "max_reward"),
        "mean_final_temperature": safe_get(summary, "mean_final_temperature"),
        "mean_final_energy": safe_get(summary, "mean_final_energy"),
        "mean_final_predicted_heat": safe_get(summary, "mean_final_predicted_heat"),
        "termination_count": safe_get(summary, "termination_count"),
        "checkpoint_path": safe_get(summary, "checkpoint_path"),
        "eval_json_path": safe_get(summary, "eval_json_path"),
    }


def save_plot_reward(df: pd.DataFrame, output_path: str) -> None:
    """
    Save a bar chart for mean reward comparison.

    Args:
        df: Comparison DataFrame.
        output_path: Output PNG file path.
    """
    ensure_dir(os.path.dirname(output_path))

    plt.figure(figsize=(10, 6))
    plt.bar(df["model"], df["mean_reward"])
    plt.title("CoolSync+ Model Comparison - Mean Reward")
    plt.xlabel("Model")
    plt.ylabel("Mean Reward")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_plot_temperature_energy(df: pd.DataFrame, temp_path: str, energy_path: str) -> None:
    """
    Save separate bar charts for temperature and energy comparison.

    Args:
        df: Comparison DataFrame.
        temp_path: Output PNG for temperature.
        energy_path: Output PNG for energy.
    """
    ensure_dir(os.path.dirname(temp_path))
    ensure_dir(os.path.dirname(energy_path))

    plt.figure(figsize=(10, 6))
    plt.bar(df["model"], df["mean_final_temperature"])
    plt.title("CoolSync+ Model Comparison - Mean Final Temperature")
    plt.xlabel("Model")
    plt.ylabel("Mean Final Temperature")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(temp_path, dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(df["model"], df["mean_final_energy"])
    plt.title("CoolSync+ Model Comparison - Mean Final Energy")
    plt.xlabel("Model")
    plt.ylabel("Mean Final Energy")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(energy_path, dpi=150)
    plt.close()


def compare_results() -> Dict[str, Any]:
    """
    Compare the main CoolSync+ RL model results using saved evaluation summaries.

    Expected files:
    - results/summaries/q_learning_eval_forecast_off.json
    - results/summaries/q_learning_eval_forecast_on.json
    - results/summaries/dqn_eval_forecast_off.json
    - results/summaries/dqn_eval_forecast_on.json

    Returns:
        Summary dictionary containing ranking and output file paths.
    """
    input_files = {
        "Q-Learning (Forecast Off)": "results/summaries/q_learning_eval_forecast_off.json",
        "Q-Learning (Forecast On)": "results/summaries/q_learning_eval_forecast_on.json",
        "DQN (Forecast Off)": "results/summaries/dqn_eval_forecast_off.json",
        "DQN (Forecast On)": "results/summaries/dqn_eval_forecast_on.json",
    }

    rows: List[Dict[str, Any]] = []
    missing_files: List[str] = []

    # Load each summary if available.
    for model_name, filepath in input_files.items():
        if not os.path.exists(filepath):
            missing_files.append(filepath)
            continue

        summary = load_json(filepath)
        row = build_result_row(
            model_name=model_name,
            summary=summary,
        )
        rows.append(row)

    if not rows:
        raise FileNotFoundError(
            "No evaluation summary files were found. "
            "Run the model evaluations first before comparing results."
        )

    # Build a DataFrame for sorting, saving, and plotting.
    df = pd.DataFrame(rows)

    # Primary ranking: higher mean reward is better.
    df_sorted_by_reward = df.sort_values(
        by="mean_reward",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    # Secondary informative views.
    df_sorted_by_temperature = df.sort_values(
        by="mean_final_temperature",
        ascending=True,
        na_position="last",
    ).reset_index(drop=True)

    df_sorted_by_energy = df.sort_values(
        by="mean_final_energy",
        ascending=True,
        na_position="last",
    ).reset_index(drop=True)

    # Output files.
    comparison_csv_path = "results/comparisons/model_comparison.csv"
    comparison_json_path = "results/comparisons/model_comparison.json"
    reward_plot_path = "results/comparisons/model_comparison_mean_reward.png"
    temp_plot_path = "results/comparisons/model_comparison_mean_temperature.png"
    energy_plot_path = "results/comparisons/model_comparison_mean_energy.png"

    ensure_dir("results/comparisons")

    # Save the main reward-ranked table.
    df_sorted_by_reward.to_csv(comparison_csv_path, index=False)

    comparison_payload = {
        "status": "comparison_complete",
        "num_models_compared": int(len(df_sorted_by_reward)),
        "missing_files": missing_files,
        "best_by_reward": df_sorted_by_reward.iloc[0]["model"] if len(df_sorted_by_reward) > 0 else None,
        "best_by_temperature": df_sorted_by_temperature.iloc[0]["model"] if len(df_sorted_by_temperature) > 0 else None,
        "best_by_energy": df_sorted_by_energy.iloc[0]["model"] if len(df_sorted_by_energy) > 0 else None,
        "comparison_csv_path": comparison_csv_path,
        "reward_plot_path": reward_plot_path,
        "temperature_plot_path": temp_plot_path,
        "energy_plot_path": energy_plot_path,
        "reward_ranking": df_sorted_by_reward.to_dict(orient="records"),
        "temperature_ranking": df_sorted_by_temperature.to_dict(orient="records"),
        "energy_ranking": df_sorted_by_energy.to_dict(orient="records"),
    }

    with open(comparison_json_path, "w", encoding="utf-8") as file:
        json.dump(comparison_payload, file, indent=2)

    # Save plots for the report.
    save_plot_reward(
        df=df_sorted_by_reward,
        output_path=reward_plot_path,
    )

    save_plot_temperature_energy(
        df=df_sorted_by_reward,
        temp_path=temp_plot_path,
        energy_path=energy_plot_path,
    )

    # Print a concise ranking summary.
    print("\n[INFO] Reward ranking:")
    for rank, row in enumerate(df_sorted_by_reward.itertuples(index=False), start=1):
        print(
            f"{rank}. {row.model} | "
            f"MeanReward={row.mean_reward} | "
            f"MeanTemp={row.mean_final_temperature} | "
            f"MeanEnergy={row.mean_final_energy} | "
            f"Terminations={row.termination_count}"
        )

    print("\n[INFO] Best models by metric:")
    print(f"Best by reward      : {comparison_payload['best_by_reward']}")
    print(f"Best by temperature : {comparison_payload['best_by_temperature']}")
    print(f"Best by energy      : {comparison_payload['best_by_energy']}")

    print("\n[INFO] Comparison files saved:")
    print(f"- CSV  : {comparison_csv_path}")
    print(f"- JSON : {comparison_json_path}")
    print(f"- Plot : {reward_plot_path}")
    print(f"- Plot : {temp_plot_path}")
    print(f"- Plot : {energy_plot_path}")

    return comparison_payload


if __name__ == "__main__":
    compare_results()