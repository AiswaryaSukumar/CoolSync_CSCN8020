# evaluate.py

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from agents.q_learning_agent import QLearningAgent
from configs.default_config import CoolSyncConfig
from envs.coolsync_env import CoolSyncEnv
from forecasting.forecast_utils import load_lstm_model
from training.state_preprocessor import StatePreprocessor
from utils.metrics import summarize_episode
from utils.seed import set_global_seed


def ensure_dir(path: str) -> None:
    """
    Create a directory if it does not already exist.
    """
    if path:
        os.makedirs(path, exist_ok=True)


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save a dictionary to JSON.
    """
    ensure_dir(os.path.dirname(filepath))

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_csv(rows: List[Dict[str, Any]], filepath: str) -> None:
    """
    Save a list of dictionaries to CSV.
    """
    if not rows:
        return

    ensure_dir(os.path.dirname(filepath))

    with open(filepath, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_q_learning(
    episodes: int = 10,
    scenario_name: str = "stable",
    use_forecast: bool = False,
    forecast_checkpoint_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate a saved Q-learning checkpoint on the CoolSync environment.
    """
    config = CoolSyncConfig()

    if seed is not None:
        config.seed = seed

    set_global_seed(config.seed)

    suffix = "with_forecast" if use_forecast else "without_forecast"
    checkpoint_path = f"results/checkpoints/q_learning_{scenario_name}_{suffix}.pkl"
    eval_csv_path = f"results/logs/q_learning_eval_{scenario_name}_{suffix}.csv"
    eval_json_path = f"results/summaries/q_learning_eval_{scenario_name}_{suffix}.json"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train the Q-learning agent first."
        )

    forecast_model = None
    forecast_device = None

    if use_forecast and forecast_checkpoint_path:
        forecast_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        forecast_model = load_lstm_model(
            checkpoint_path=forecast_checkpoint_path,
            device=forecast_device,
        )

    preprocessor = StatePreprocessor(
        config=config,
        use_forecast=use_forecast,
    )

    agent = QLearningAgent(
        preprocessor=preprocessor,
        num_actions=3,
    )
    agent.load_q_table(checkpoint_path)
    agent.set_eval_mode()

    episode_logs: List[Dict[str, Any]] = []

    for episode in range(1, episodes + 1):
        env = CoolSyncEnv(
            config=config,
            scenario_name=scenario_name,
            use_forecast=use_forecast,
            forecast_model=forecast_model,
            forecast_device=forecast_device,
        )

        state, _ = env.reset()

        terminated = False
        truncated = False
        episode_reward = 0.0

        while not (terminated or truncated):
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)

            state = next_state
            episode_reward += float(reward)

        episode_summary = summarize_episode(
            episode_history=env.history,
            safe_temp_max=config.safe_temp_max,
        )

        episode_log = {
            "episode": episode,
            "scenario_name": scenario_name,
            "use_forecast": int(use_forecast),
            "episode_reward": float(episode_reward),
            **episode_summary,
        }
        episode_logs.append(episode_log)

        print(
            f"[Q-Learning Evaluation] Episode {episode}/{episodes} | "
            f"Reward={episode_reward:.3f} | "
            f"Energy={episode_summary['total_energy']:.3f} | "
            f"MaxTemp={episode_summary['max_temperature']:.3f}"
        )

        env.close()

    save_csv(episode_logs, eval_csv_path)

    rewards = [row["episode_reward"] for row in episode_logs]

    summary = {
        "status": "evaluation_complete",
        "scenario_name": scenario_name,
        "use_forecast": use_forecast,
        "checkpoint_path": checkpoint_path,
        "eval_csv_path": eval_csv_path,
        "eval_json_path": eval_json_path,
        "episodes_evaluated": episodes,
        "mean_reward": float(np.mean(rewards)) if rewards else None,
        "std_reward": float(np.std(rewards)) if rewards else None,
        "min_reward": float(np.min(rewards)) if rewards else None,
        "max_reward": float(np.max(rewards)) if rewards else None,
        "mean_total_energy": float(np.mean([row["total_energy"] for row in episode_logs])),
        "mean_avg_temperature": float(np.mean([row["avg_temperature"] for row in episode_logs])),
        "mean_max_temperature": float(np.mean([row["max_temperature"] for row in episode_logs])),
        "mean_overheat_count": float(np.mean([row["overheat_count"] for row in episode_logs])),
    }

    save_json(summary, eval_json_path)

    print("\n[INFO] Q-learning evaluation summary:")
    print(json.dumps(summary, indent=2))

    return summary


if __name__ == "__main__":
    evaluate_q_learning(
        episodes=10,
        scenario_name="stable",
        use_forecast=False,
    )