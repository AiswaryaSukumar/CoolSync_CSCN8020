# train_dqn.py

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

from agents.dqn_agent import DQNAgent
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


def train_dqn(
    episodes: int = 250,
    scenario_name: str = "stable",
    use_forecast: bool = True,
    max_steps_per_episode: Optional[int] = None,
    forecast_checkpoint_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Train DQN on the CoolSync environment.

    Args:
        episodes: Number of training episodes
        scenario_name: Scenario to train on
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

    if seed is not None:
        config.seed = seed

    set_global_seed(config.seed)

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

    sample_state, _ = env.reset()
    state_dim = len(preprocessor.normalize_state(sample_state))

    # ------------------------------------------------------------------
    # DQN agent
    # ------------------------------------------------------------------
    agent = DQNAgent(
        preprocessor=preprocessor,
        state_dim=state_dim,
        num_actions=env.action_space.n,
        learning_rate=1e-3,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        replay_capacity=10000,
        target_update_freq=10,
    )

    print(
        f"[DQN] Starting training | "
        f"Scenario={scenario_name} | "
        f"Forecast={'ON' if use_forecast else 'OFF'} | "
        f"Episodes={episodes} | "
        f"EpisodeLength={config.episode_length} | "
        f"Device={agent.device}"
    )

    # ------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------
    suffix = "with_forecast" if use_forecast else "without_forecast"

    logs_csv_path = f"results/logs/dqn_{scenario_name}_{suffix}.csv"
    logs_json_path = f"results/logs/dqn_{scenario_name}_{suffix}.json"
    checkpoint_path = f"results/checkpoints/dqn_{scenario_name}_{suffix}.pth"

    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/checkpoints", exist_ok=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    training_logs = []

    best_episode_reward = float("-inf")
    best_episode_index = 0

    for episode in range(1, episodes + 1):
        state, _ = env.reset()

        terminated = False
        truncated = False
        episode_reward = 0.0
        episode_losses = []
        episode_steps = 0

        while not (terminated or truncated):
            # Select action with epsilon-greedy policy
            action = agent.select_action(state, training=True)

            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            # Store transition in replay buffer
            agent.store_transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )

            # Perform one gradient update
            train_info = agent.train_step()
            episode_losses.append(train_info["loss"])

            state = next_state
            episode_reward += reward
            episode_steps += 1

        # Update target network periodically
        agent.maybe_update_target_network(episode=episode)

        # Decay exploration
        agent.decay_epsilon()

        episode_summary = summarize_episode(
            episode_history=env.history,
            safe_temp_max=config.safe_temp_max,
        )

        avg_loss = (
            sum(episode_losses) / len(episode_losses) if episode_losses else 0.0
        )

        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            best_episode_index = episode

        append_training_log(
            training_logs,
            {
                "episode": episode,
                "scenario_name": scenario_name,
                "use_forecast": int(use_forecast),
                "episode_steps": int(episode_steps),
                "episode_reward": float(episode_reward),
                "epsilon": float(agent.epsilon),
                "avg_loss": float(avg_loss),
                "replay_buffer_size": int(len(agent.replay_buffer)),
                "best_episode_reward_so_far": float(best_episode_reward),
                **episode_summary,
            },
        )

        if episode % 10 == 0 or episode == 1 or episode == episodes:
            print(
                f"[DQN] Episode {episode}/{episodes} | "
                f"Reward={episode_reward:.2f} | "
                f"Epsilon={agent.epsilon:.3f} | "
                f"AvgLoss={avg_loss:.5f} | "
                f"Steps={episode_steps} | "
                f"Buffer={len(agent.replay_buffer)} | "
                f"Energy={episode_summary['total_energy']:.2f} | "
                f"MaxTemp={episode_summary['max_temperature']:.2f}"
            )

    # ------------------------------------------------------------------
    # Save logs and checkpoint
    # ------------------------------------------------------------------
    save_logs_to_csv(training_logs, logs_csv_path)
    save_logs_to_json(training_logs, logs_json_path)
    agent.save_checkpoint(checkpoint_path)

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
        "final_replay_buffer_size": int(len(agent.replay_buffer)),
        "logs_csv_path": logs_csv_path,
        "logs_json_path": logs_json_path,
        "checkpoint_path": checkpoint_path,
        "device": str(agent.device),
    }

    print(
        f"[DQN] Training complete | "
        f"BestReward={best_episode_reward:.2f} (Episode {best_episode_index}) | "
        f"FinalEpsilon={agent.epsilon:.3f} | "
        f"ReplayBufferSize={len(agent.replay_buffer)}"
    )

    env.close()

    return {
        "agent_summary": agent.summary(),
        "final_summary": final_summary,
        "training_logs": training_logs,
    }


if __name__ == "__main__":
    train_dqn()