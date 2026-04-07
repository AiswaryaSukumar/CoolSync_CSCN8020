from __future__ import annotations

import argparse
from typing import Optional

from compare_controllers import compare_controllers
from configs.default_config import CoolSyncConfig
from forecasting.evaluate_lstm import evaluate_lstm
from forecasting.trace_generation import (
    DEFAULT_TRACE_SCENARIOS,
    generate_and_save_trace_dataset,
)
from forecasting.train_lstm import train_lstm
from run_experiments import run_experiments
from train_dqn import train_dqn
from train_q_learning import train_q_learning


DEFAULT_LSTM_CHECKPOINT = "results/checkpoints/lstm_heat_predictor_best.pth"


def generate_prompt_driven_data(
    steps_per_scenario: int = 600,
    seed: Optional[int] = None,
) -> dict:
    """
    Generate the prompt-driven trace dataset used by the LSTM pipeline.
    """
    config = CoolSyncConfig()

    if seed is not None:
        config.seed = seed

    summary = generate_and_save_trace_dataset(
        config=config,
        scenarios=DEFAULT_TRACE_SCENARIOS,
        steps_per_scenario=steps_per_scenario,
        output_path="data/generated/prompt_driven_heat_traces.csv",
    )

    print("\n=== Prompt-driven dataset generated ===")
    print(f"Rows      : {summary['num_rows']}")
    print(f"Scenarios : {', '.join(summary['scenarios'])}")
    print(f"Saved to  : {summary['output_path']}")

    return summary


def run_full_pipeline(
    rl_episodes: int = 100,
    compare_episodes: int = 5,
    experiment_episodes: int = 5,
    steps_per_scenario: int = 600,
    seed: Optional[int] = None,
) -> None:
    """
    Run the full project pipeline in skeleton-compliant order.
    """
    print("\n=== Step 1: Generate prompt-driven traces ===")
    generate_prompt_driven_data(
        steps_per_scenario=steps_per_scenario,
        seed=seed,
    )

    print("\n=== Step 2: Train LSTM ===")
    train_lstm(
        steps_per_scenario=steps_per_scenario,
    )

    print("\n=== Step 3: Evaluate LSTM ===")
    evaluate_lstm()

    print("\n=== Step 4: Train Q-learning (without forecast) ===")
    train_q_learning(
        episodes=rl_episodes,
        scenario_name="stable",
        use_forecast=False,
        seed=seed,
    )

    print("\n=== Step 5: Train Q-learning (with forecast) ===")
    train_q_learning(
        episodes=rl_episodes,
        scenario_name="stable",
        use_forecast=True,
        forecast_checkpoint_path=DEFAULT_LSTM_CHECKPOINT,
        seed=seed,
    )

    print("\n=== Step 6: Train DQN (without forecast) ===")
    train_dqn(
        episodes=rl_episodes,
        scenario_name="stable",
        use_forecast=False,
        seed=seed,
    )

    print("\n=== Step 7: Train DQN (with forecast) ===")
    train_dqn(
        episodes=rl_episodes,
        scenario_name="stable",
        use_forecast=True,
        forecast_checkpoint_path=DEFAULT_LSTM_CHECKPOINT,
        seed=seed,
    )

    print("\n=== Step 8: Compare controllers on stable scenario ===")
    compare_controllers(
        scenario_name="stable",
        episodes=compare_episodes,
        include_rl=True,
        sort_by_reward=True,
    )

    print("\n=== Step 9: Run full scenario sweep ===")
    run_experiments(
        scenarios=["stable", "sinusoidal", "spiky", "burst_heavy"],
        episodes_per_controller=experiment_episodes,
        include_rl=True,
        sort_by_reward=True,
    )

    print("\n=== Full pipeline complete ===")


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser for flexible execution.
    """
    parser = argparse.ArgumentParser(description="CoolSync+ main runner")

    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=[
            "full",
            "generate_data",
            "train_lstm",
            "evaluate_lstm",
            "train_q_learning",
            "train_dqn",
            "compare",
            "experiments",
        ],
        help="Which part of the pipeline to run.",
    )

    parser.add_argument(
        "--use_forecast",
        action="store_true",
        help="Enable forecast-aware RL for train_q_learning or train_dqn modes.",
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default="stable",
        help="Scenario name for compare or RL training modes.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes for compare/experiments modes or direct training modes.",
    )

    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=["stable", "sinusoidal", "spiky", "burst_heavy"],
        help="Scenario list for experiments mode.",
    )

    parser.add_argument(
        "--steps_per_scenario",
        type=int,
        default=600,
        help="Number of generated trace steps per scenario for data/LSTM generation.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed override.",
    )

    parser.add_argument(
        "--forecast_checkpoint_path",
        type=str,
        default=DEFAULT_LSTM_CHECKPOINT,
        help="Path to trained LSTM checkpoint for forecast-enabled RL training.",
    )

    return parser


def main() -> None:
    """
    Entry point for the CoolSync+ project.
    """
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "full":
        run_full_pipeline(
            rl_episodes=100,
            compare_episodes=5,
            experiment_episodes=5,
            steps_per_scenario=args.steps_per_scenario,
            seed=args.seed,
        )

    elif args.mode == "generate_data":
        generate_prompt_driven_data(
            steps_per_scenario=args.steps_per_scenario,
            seed=args.seed,
        )

    elif args.mode == "train_lstm":
        train_lstm(
            steps_per_scenario=args.steps_per_scenario,
        )

    elif args.mode == "evaluate_lstm":
        evaluate_lstm(
            checkpoint_path=args.forecast_checkpoint_path,
        )

    elif args.mode == "train_q_learning":
        train_q_learning(
            episodes=args.episodes,
            scenario_name=args.scenario,
            use_forecast=args.use_forecast,
            forecast_checkpoint_path=(
                args.forecast_checkpoint_path if args.use_forecast else None
            ),
            seed=args.seed,
        )

    elif args.mode == "train_dqn":
        train_dqn(
            episodes=args.episodes,
            scenario_name=args.scenario,
            use_forecast=args.use_forecast,
            forecast_checkpoint_path=(
                args.forecast_checkpoint_path if args.use_forecast else None
            ),
            seed=args.seed,
        )

    elif args.mode == "compare":
        compare_controllers(
            scenario_name=args.scenario,
            episodes=args.episodes,
            include_rl=True,
            sort_by_reward=True,
        )

    elif args.mode == "experiments":
        run_experiments(
            scenarios=args.scenarios,
            episodes_per_controller=args.episodes,
            include_rl=True,
            sort_by_reward=True,
        )


if __name__ == "__main__":
    main()