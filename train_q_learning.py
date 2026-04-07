# train_q_learning.py

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

from agents.q_learning_agent import QLearningAgent
from configs.default_config import CoolSyncConfig
from envs.coolsync_env import CoolSyncEnv
from forecasting.forecast_utils import load_lstm_model
from training.state_preprocessor import StatePreprocessor
from training.train_utils import (
    append_training_log,
    save_logs_to_csv,
    save_logs_to_json,
)
from utils.metrics import summarize_episode
from utils.seed import set_global_seed


def train_q_learning(
    episodes: int = 200,
    scenario_name: str = "stable",
    use_forecast: bool = True,
    max_steps_per_episode: Optional[int] = None,
    forecast_checkpoint_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Train a tabular Q-learning agent on the CoolSync environment.

    Args:
        episodes: Number of training episodes
        scenario_name: Scenario to train on (e.g., stable, sinusoidal, spiky, burst_heavy)
        use_forecast: Whether forecast-enhanced state should be used
        max_steps_per_episode: Optional override for episode length
        forecast_checkpoint_path: Optional LSTM checkpoint for forecast-enabled training
        seed: Optional random seed override

    Returns:
        Dictionary containing training artifacts and output file paths
    """
    # ------------------------------------------------------------------
    # Configuration and reproducibility
    # ------------------------------------------------------------------
    config = CoolSyncConfig()

    # Override config seed if provided by caller
    if seed is not None:
        config.seed = seed

    # Set global seed for reproducibility
    set_global_seed(config.seed)

    # Allow optional episode length override without changing the config class
    if max_steps_per_episode is not None:
        config.episode_length = max_steps_per_episode

    # ------------------------------------------------------------------
    # Optional forecast model loading
    # ------------------------------------------------------------------
    forecast_model = None
    forecast_device = None

    if use_forecast and forecast_checkpoint_path:
        forecast_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        forecast_model = load_lstm_model(
            checkpoint_path=forecast_checkpoint_path,
            device=forecast_device,
        )

    # ------------------------------------------------------------------
    # Environment and preprocessing
    # ------------------------------------------------------------------
    env = CoolSyncEnv(
        config=config,
        scenario_name=scenario_name,
        use_forecast=use_forecast,
        forecast_model=forecast_model,
        forecast_device=forecast_device,
    )

    preprocessor = StatePreprocessor(
        config=config,
        use_forecast=use_forecast,
    )

    # ------------------------------------------------------------------
    # Q-learning agent
    # ------------------------------------------------------------------
    agent = QLearningAgent(
        preprocessor=preprocessor,
        num_actions=env.action_space.n,
        learning_rate=0.10,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    )

    # ------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------
    suffix = "with_forecast" if use_forecast else "without_forecast"

    logs_csv_path = f"results/logs/q_learning_{scenario_name}_{suffix}.csv"
    logs_json_path = f"results/logs/q_learning_{scenario_name}_{suffix}.json"
    checkpoint_path = f"results/checkpoints/q_learning_{scenario_name}_{suffix}.pkl"

    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/checkpoints", exist_ok=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    training_logs = []

    best_episode_reward = float("-inf")
    best_episode_index = 0

    print(
        f"[Q-Learning] Starting training | "
        f"Scenario={scenario_name} | "
        f"Forecast={'ON' if use_forecast else 'OFF'} | "
        f"Episodes={episodes} | "
        f"EpisodeLength={config.episode_length}"
    )

    for episode in range(1, episodes + 1):
        # Reset environment at the start of each episode
        state, _ = env.reset()

        terminated = False
        truncated = False
        episode_reward = 0.0
        episode_steps = 0

        # Run one full episode
        while not (terminated or truncated):
            # Select action via epsilon-greedy exploration
            action = agent.select_action(state, training=True)

            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)

            # Treat either termination or truncation as episode end
            done = terminated or truncated

            # Apply Bellman Q-learning update
            agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )

            # Advance state
            state = next_state
            episode_reward += reward
            episode_steps += 1

        # Decay exploration after each episode
        agent.decay_epsilon()

        # Summarize the finished episode using the stored environment history
        episode_summary = summarize_episode(
            episode_history=env.history,
            safe_temp_max=config.safe_temp_max,
        )

        # Track best reward achieved so far
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            best_episode_index = episode

        # Append episode log row
        append_training_log(
            training_logs,
            {
                "episode": episode,
                "scenario_name": scenario_name,
                "use_forecast": int(use_forecast),
                "episode_steps": int(episode_steps),
                "episode_reward": float(episode_reward),
                "epsilon": float(agent.epsilon),
                "q_table_size": int(agent.get_q_table_size()),
                "best_episode_reward_so_far": float(best_episode_reward),
                **episode_summary,
            },
        )

        # Periodic console progress
        if episode % 10 == 0 or episode == 1 or episode == episodes:
            print(
                f"[Q-Learning] Episode {episode}/{episodes} | "
                f"Reward={episode_reward:.2f} | "
                f"Epsilon={agent.epsilon:.3f} | "
                f"Steps={episode_steps} | "
                f"QTableSize={agent.get_q_table_size()} | "
                f"Energy={episode_summary['total_energy']:.2f} | "
                f"MaxTemp={episode_summary['max_temperature']:.2f}"
            )

    # ------------------------------------------------------------------
    # Save logs and learned Q-table
    # ------------------------------------------------------------------
    save_logs_to_csv(training_logs, logs_csv_path)
    save_logs_to_json(training_logs, logs_json_path)
    agent.save_q_table(checkpoint_path)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    final_summary = {
        "episodes": episodes,
        "scenario_name": scenario_name,
        "use_forecast": use_forecast,
        "seed": config.seed,
        "best_episode_reward": float(best_episode_reward),
        "best_episode_index": int(best_episode_index),
        "final_epsilon": float(agent.epsilon),
        "final_q_table_size": int(agent.get_q_table_size()),
        "logs_csv_path": logs_csv_path,
        "logs_json_path": logs_json_path,
        "checkpoint_path": checkpoint_path,
    }

    print(
        f"[Q-Learning] Training complete | "
        f"BestReward={best_episode_reward:.2f} (Episode {best_episode_index}) | "
        f"FinalEpsilon={agent.epsilon:.3f} | "
        f"FinalQTableSize={agent.get_q_table_size()}"
    )

    env.close()

    return {
        "agent_summary": agent.summary(),
        "final_summary": final_summary,
        "training_logs": training_logs,
    }


if __name__ == "__main__":
    train_q_learning()