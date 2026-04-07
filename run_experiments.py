# run_experiments.py

from __future__ import annotations

import os
from typing import Dict, List

from compare_controllers import compare_controllers
from utils.logger import save_csv_rows, save_json


DEFAULT_SCENARIOS = [
    "stable",
    "sinusoidal",
    "spiky",
    "burst_heavy",
]


def ensure_results_dirs() -> None:
    """
    Ensure output directories exist.
    """
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/summaries", exist_ok=True)


def run_experiments(
    scenarios: List[str] | None = None,
    episodes_per_controller: int = 5,
    include_rl: bool = True,
    sort_by_reward: bool = True,
) -> Dict:
    """
    Run controller comparisons across all requested scenarios.

    This scenario sweep runner calls compare_controllers(...) once per scenario
    and aggregates the results into single CSV + JSON summaries.
    """
    ensure_results_dirs()

    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS

    all_rows: List[Dict] = []
    scenario_summaries: List[Dict] = []

    for scenario_name in scenarios:
        print(f"\n=== Running scenario: {scenario_name} ===")

        scenario_summary = compare_controllers(
            scenario_name=scenario_name,
            episodes=episodes_per_controller,
            include_rl=include_rl,
            sort_by_reward=sort_by_reward,
        )

        scenario_summaries.append(scenario_summary)

        for row in scenario_summary["results"]:
            # Add explicit scenario field again for clarity in aggregate tables
            aggregate_row = dict(row)
            aggregate_row["scenario_name"] = scenario_name
            all_rows.append(aggregate_row)

    # Optional aggregate ranking by scenario + reward
    if sort_by_reward:
        # Keep row order grouped by scenario, then reward rank if already present
        all_rows = sorted(
            all_rows,
            key=lambda row: (
                row.get("scenario_name", ""),
                row.get("rank_by_mean_total_reward", 9999),
            ),
        )

    aggregate_csv_path = "results/logs/experiment_sweep_all_scenarios.csv"
    aggregate_json_path = "results/summaries/experiment_sweep_all_scenarios.json"

    save_csv_rows(
        rows=all_rows,
        filepath=aggregate_csv_path,
    )

    final_summary = {
        "scenarios": scenarios,
        "episodes_per_controller": episodes_per_controller,
        "include_rl": include_rl,
        "sort_by_reward": sort_by_reward,
        "num_total_rows": len(all_rows),
        "scenario_summaries": scenario_summaries,
        "aggregate_csv_path": aggregate_csv_path,
        "aggregate_json_path": aggregate_json_path,
    }

    save_json(
        data=final_summary,
        filepath=aggregate_json_path,
    )

    return final_summary


if __name__ == "__main__":
    summary = run_experiments(
        scenarios=DEFAULT_SCENARIOS,
        episodes_per_controller=5,
        include_rl=True,
        sort_by_reward=True,
    )

    print("\nScenario sweep complete.")
    print(f"Saved aggregate CSV : {summary['aggregate_csv_path']}")
    print(f"Saved aggregate JSON: {summary['aggregate_json_path']}")