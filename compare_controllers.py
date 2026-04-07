# compare_controllers.py

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch

from agents.dqn_agent import DQNAgent
from agents.q_learning_agent import QLearningAgent
from baselines.pid_controller import SimplePIDCoolingController
from baselines.predictive_threshold_controller import PredictiveThresholdCoolingController
from baselines.static_controller import StaticCoolingController
from baselines.threshold_controller import ThresholdCoolingController
from configs.default_config import CoolSyncConfig
from envs.coolsync_env import CoolSyncEnv
from forecasting.forecast_utils import load_lstm_model
from training.state_preprocessor import StatePreprocessor
from utils.logger import save_csv_rows, save_json
from utils.metrics import summarize_episode
from utils.seed import set_global_seed


def ensure_results_dirs() -> None:
    """
    Ensure required result directories exist.
    """
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/summaries", exist_ok=True)


def resolve_forecast_checkpoint_path(
    checkpoint_path: str = "results/checkpoints/lstm_heat_predictor_best.pth",
) -> Optional[str]:
    """
    Resolve the preferred LSTM checkpoint path while supporting common fallback names.
    """
    candidate_paths = [
        checkpoint_path,
        "results/checkpoints/lstm_best.pth",
        "results/checkpoints/lstm_smoke_test.pth",
    ]

    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path

    return None


def build_forecast_artifacts(
    config: CoolSyncConfig,
    use_forecast: bool,
    checkpoint_path: str = "results/checkpoints/lstm_heat_predictor_best.pth",
) -> Tuple[Optional[Any], Optional[torch.device], Optional[str]]:
    """
    Build forecast model and device if forecast is enabled and checkpoint exists.

    Returns:
        forecast_model, device, resolved_checkpoint_path
    """
    if not use_forecast:
        return None, None, None

    resolved_checkpoint_path = resolve_forecast_checkpoint_path(checkpoint_path)
    if resolved_checkpoint_path is None:
        return None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forecast_model = load_lstm_model(
        checkpoint_path=resolved_checkpoint_path,
        device=device,
    )
    return forecast_model, device, resolved_checkpoint_path


def resolve_rl_checkpoint_path(
    algorithm_name: str,
    scenario_name: str,
    use_forecast: bool,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve the best available RL checkpoint.

    Preference order:
    1. scenario-specific checkpoint
    2. stable checkpoint as fallback
    """
    suffix = "with_forecast" if use_forecast else "without_forecast"
    extension = "pkl" if algorithm_name == "q_learning" else "pth"

    candidate_paths = [
        (
            f"results/checkpoints/{algorithm_name}_{scenario_name}_{suffix}.{extension}",
            scenario_name,
        ),
        (
            f"results/checkpoints/{algorithm_name}_stable_{suffix}.{extension}",
            "stable",
        ),
    ]

    for candidate_path, source_scenario in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path, source_scenario

    return None, None


def build_env(
    config: CoolSyncConfig,
    scenario_name: str,
    use_forecast: bool,
    forecast_model=None,
    forecast_device=None,
) -> CoolSyncEnv:
    """
    Build a CoolSync environment for a single controller run.
    """
    return CoolSyncEnv(
        config=config,
        scenario_name=scenario_name,
        use_forecast=use_forecast,
        forecast_model=forecast_model,
        forecast_device=forecast_device,
    )


def load_q_learning_agent(
    config: CoolSyncConfig,
    checkpoint_path: str,
    use_forecast: bool,
) -> Optional[QLearningAgent]:
    """
    Load a trained Q-learning agent from disk.
    """
    if not os.path.exists(checkpoint_path):
        return None

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

    return agent


def load_dqn_agent(
    config: CoolSyncConfig,
    checkpoint_path: str,
    use_forecast: bool,
) -> Optional[DQNAgent]:
    """
    Load a trained DQN agent from disk.
    """
    if not os.path.exists(checkpoint_path):
        return None

    preprocessor = StatePreprocessor(
        config=config,
        use_forecast=use_forecast,
    )

    state_dim = 6 if use_forecast else 5

    agent = DQNAgent(
        preprocessor=preprocessor,
        state_dim=state_dim,
        num_actions=3,
    )
    agent.load_checkpoint(checkpoint_path)
    agent.set_eval_mode()

    return agent


def select_action_for_controller(controller, state, is_rl: bool) -> int:
    """
    Select an action from either a baseline or RL controller.
    """
    if is_rl:
        return int(controller.select_action(state, training=False))
    return int(controller.select_action(state))


def run_single_episode(
    env: CoolSyncEnv,
    controller,
    is_rl: bool = False,
) -> Dict[str, Any]:
    """
    Run one evaluation episode and summarize results.
    """
    state, _ = env.reset()

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = select_action_for_controller(
            controller=controller,
            state=state,
            is_rl=is_rl,
        )
        state, reward, terminated, truncated, info = env.step(action)

    summary = summarize_episode(
        episode_history=env.history,
        safe_temp_max=env.config.safe_temp_max,
    )

    return summary


def evaluate_controller(
    controller_name: str,
    controller,
    config: CoolSyncConfig,
    scenario_name: str,
    episodes: int,
    use_forecast: bool,
    forecast_model=None,
    forecast_device=None,
    is_rl: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a controller across multiple episodes.
    """
    episode_summaries: List[Dict[str, Any]] = []

    for _ in range(episodes):
        env = build_env(
            config=config,
            scenario_name=scenario_name,
            use_forecast=use_forecast,
            forecast_model=forecast_model,
            forecast_device=forecast_device,
        )

        episode_summary = run_single_episode(
            env=env,
            controller=controller,
            is_rl=is_rl,
        )
        episode_summaries.append(episode_summary)
        env.close()

    df = pd.DataFrame(episode_summaries)

    return {
        "controller": controller_name,
        "scenario_name": scenario_name,
        "episodes": episodes,
        "use_forecast": int(use_forecast),
        "mean_total_reward": float(df["total_reward"].mean()),
        "std_total_reward": float(df["total_reward"].std(ddof=0)),
        "mean_total_energy": float(df["total_energy"].mean()),
        "mean_avg_temperature": float(df["avg_temperature"].mean()),
        "mean_max_temperature": float(df["max_temperature"].mean()),
        "mean_overheat_count": float(df["overheat_count"].mean()),
        "mean_overcool_count": float(df["overcool_count"].mean()),
        "mean_cooling_variance": float(df["cooling_variance"].mean()),
        "mean_recovery_time": float(df["recovery_time"].mean()),
    }


def maybe_evaluate_rl_controller(
    algorithm_name: str,
    controller_name: str,
    config: CoolSyncConfig,
    scenario_name: str,
    episodes: int,
    use_forecast: bool,
    forecast_model=None,
    forecast_device=None,
) -> Optional[Dict[str, Any]]:
    """
    Resolve checkpoint, load RL controller, and evaluate it if available.
    """
    checkpoint_path, source_scenario = resolve_rl_checkpoint_path(
        algorithm_name=algorithm_name,
        scenario_name=scenario_name,
        use_forecast=use_forecast,
    )

    if checkpoint_path is None:
        return None

    if algorithm_name == "q_learning":
        controller = load_q_learning_agent(
            config=config,
            checkpoint_path=checkpoint_path,
            use_forecast=use_forecast,
        )
    elif algorithm_name == "dqn":
        controller = load_dqn_agent(
            config=config,
            checkpoint_path=checkpoint_path,
            use_forecast=use_forecast,
        )
    else:
        raise ValueError(f"Unsupported algorithm_name: {algorithm_name}")

    if controller is None:
        return None

    row = evaluate_controller(
        controller_name=controller_name,
        controller=controller,
        config=config,
        scenario_name=scenario_name,
        episodes=episodes,
        use_forecast=use_forecast,
        forecast_model=forecast_model,
        forecast_device=forecast_device,
        is_rl=True,
    )
    row["policy_source_scenario"] = source_scenario
    row["checkpoint_path"] = checkpoint_path

    return row


def compare_controllers(
    scenario_name: str = "stable",
    episodes: int = 10,
    include_rl: bool = True,
    sort_by_reward: bool = True,
) -> Dict[str, Any]:
    """
    Compare baseline and RL controllers on one scenario.

    Controllers compared:
    - static
    - threshold
    - pid
    - predictive_threshold
    - q_learning_without_forecast
    - q_learning_with_forecast
    - dqn_without_forecast
    - dqn_with_forecast
    """
    ensure_results_dirs()

    config = CoolSyncConfig()
    set_global_seed(config.seed)

    results: List[Dict[str, Any]] = []

    # Build forecast artifacts once and reuse for forecast-enabled evaluations
    forecast_model, forecast_device, forecast_checkpoint_path = build_forecast_artifacts(
        config=config,
        use_forecast=True,
    )

    # --------------------------------------------------------------
    # Baseline controllers
    # --------------------------------------------------------------
    static_controller = StaticCoolingController(fixed_action=1)
    threshold_controller = ThresholdCoolingController(
        safe_temp_min=config.safe_temp_min,
        safe_temp_max=config.safe_temp_max,
    )
    pid_controller = SimplePIDCoolingController(target_temp=24.0)
    predictive_threshold_controller = PredictiveThresholdCoolingController(
        safe_temp_min=config.safe_temp_min,
        safe_temp_max=config.safe_temp_max,
        predicted_heat_threshold=0.75,
        low_heat_threshold=0.25,
    )

    results.append(
        evaluate_controller(
            controller_name="static",
            controller=static_controller,
            config=config,
            scenario_name=scenario_name,
            episodes=episodes,
            use_forecast=False,
            is_rl=False,
        )
    )

    results.append(
        evaluate_controller(
            controller_name="threshold",
            controller=threshold_controller,
            config=config,
            scenario_name=scenario_name,
            episodes=episodes,
            use_forecast=False,
            is_rl=False,
        )
    )

    results.append(
        evaluate_controller(
            controller_name="pid",
            controller=pid_controller,
            config=config,
            scenario_name=scenario_name,
            episodes=episodes,
            use_forecast=False,
            is_rl=False,
        )
    )

    results.append(
        evaluate_controller(
            controller_name="predictive_threshold",
            controller=predictive_threshold_controller,
            config=config,
            scenario_name=scenario_name,
            episodes=episodes,
            use_forecast=True,
            forecast_model=forecast_model,
            forecast_device=forecast_device,
            is_rl=False,
        )
    )

    # --------------------------------------------------------------
    # RL controllers
    # --------------------------------------------------------------
    if include_rl:
        rl_specs = [
            ("q_learning", "q_learning_without_forecast", False),
            ("q_learning", "q_learning_with_forecast", True),
            ("dqn", "dqn_without_forecast", False),
            ("dqn", "dqn_with_forecast", True),
        ]

        for algorithm_name, controller_name, use_forecast in rl_specs:
            row = maybe_evaluate_rl_controller(
                algorithm_name=algorithm_name,
                controller_name=controller_name,
                config=config,
                scenario_name=scenario_name,
                episodes=episodes,
                use_forecast=use_forecast,
                forecast_model=forecast_model if use_forecast else None,
                forecast_device=forecast_device if use_forecast else None,
            )
            if row is not None:
                results.append(row)

    # --------------------------------------------------------------
    # Optional ranking
    # --------------------------------------------------------------
    if sort_by_reward:
        results = sorted(
            results,
            key=lambda row: row["mean_total_reward"],
            reverse=True,
        )

        # Assign rank after sorting
        for rank, row in enumerate(results, start=1):
            row["rank_by_mean_total_reward"] = rank

    # --------------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------------
    results_csv_path = f"results/logs/controller_comparison_{scenario_name}.csv"
    results_json_path = f"results/summaries/controller_comparison_{scenario_name}.json"

    save_csv_rows(
        rows=results,
        filepath=results_csv_path,
    )

    summary = {
        "scenario_name": scenario_name,
        "episodes_per_controller": episodes,
        "include_rl": include_rl,
        "forecast_checkpoint_path": forecast_checkpoint_path,
        "num_results": len(results),
        "results_csv_path": results_csv_path,
        "results_json_path": results_json_path,
        "results": results,
    }

    save_json(
        data=summary,
        filepath=results_json_path,
    )

    return summary


if __name__ == "__main__":
    summary = compare_controllers(
        scenario_name="stable",
        episodes=5,
        include_rl=True,
        sort_by_reward=True,
    )

    print("\nController comparison complete.\n")
    for row in summary["results"]:
        print(
            f"{row.get('rank_by_mean_total_reward', '-')}. "
            f"{row['controller']}: "
            f"reward={row['mean_total_reward']:.3f}, "
            f"energy={row['mean_total_energy']:.3f}, "
            f"max_temp={row['mean_max_temperature']:.3f}, "
            f"overheat={row['mean_overheat_count']:.3f}"
        )