from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from configs.default_config import CoolSyncConfig
from prompt_model.prompt_generator import generate_prompt_features
from prompt_model.prompt_to_heat import prompt_to_workload_and_heat
from scenarios.scenario_definitions import SCENARIOS
from utils.logger import ensure_parent_dir
from utils.seed import set_global_seed


DEFAULT_TRACE_SCENARIOS: List[str] = [
    "stable",
    "sinusoidal",
    "spiky",
    "burst_heavy",
]


def scenario_temporal_modifier(scenario_name: str, step: int) -> float:
    """
    Return a scenario-specific temporal heat modifier.

    This creates clearer time-series signatures that the LSTM can learn:
    - stable: near-flat behavior
    - sinusoidal: smooth oscillation
    - spiky: periodic short peaks
    - burst_heavy: wider burst windows
    """
    if scenario_name == "stable":
        return 0.0

    if scenario_name == "sinusoidal":
        return 0.10 * math.sin((2.0 * math.pi * step) / 40.0)

    if scenario_name == "spiky":
        return 0.22 if (step % 35) in {0, 1, 2} else 0.0

    if scenario_name == "burst_heavy":
        return 0.28 if (step % 24) < 5 else 0.03

    return 0.0


def generate_prompt_trace_rows(
    config: CoolSyncConfig,
    scenario_name: str,
    num_steps: int = 600,
    seed: int | None = None,
) -> List[Dict]:
    """
    Generate a prompt-driven trace with one row per time step.

    Each row contains prompt features plus generated workload and heat.
    Heat is smoothed over time so forecasting becomes a real sequential task.
    """
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario_name: {scenario_name}")

    if seed is not None:
        set_global_seed(seed)

    scenario_config = SCENARIOS[scenario_name]
    previous_heat = 0.0
    rows: List[Dict] = []

    for step in range(num_steps):
        # Generate one structured prompt event according to scenario-specific probabilities
        prompt_features = generate_prompt_features(
            config=config,
            prompt_type_probabilities=scenario_config["prompt_type_probabilities"],
        )

        # Convert prompt features into workload and instantaneous heat
        prompt_event = prompt_to_workload_and_heat(
            prompt_features=prompt_features,
            config=config,
        )

        instantaneous_heat = float(prompt_event["heat_load"])

        # Blend current heat with previous heat to inject temporal memory
        smoothed_heat = (
            0.70 * instantaneous_heat
            + 0.20 * previous_heat
            + scenario_temporal_modifier(scenario_name=scenario_name, step=step)
        )

        smoothed_heat = float(
            np.clip(smoothed_heat, config.min_heat_load, config.max_heat_load)
        )

        # Keep workload aligned with final heat trace while preserving explicit workload from prompt model
        workload = float(
            np.clip(
                0.70 * float(prompt_event["workload"])
                + 0.30 * smoothed_heat * config.workload_from_heat_scale,
                config.min_workload,
                config.max_workload,
            )
        )

        rows.append(
            {
                "scenario_name": scenario_name,
                "step": step,
                "prompt_type": prompt_event["prompt_type"],
                "prompt_length": int(prompt_event["prompt_length"]),
                "complexity_score": float(prompt_event["complexity_score"]),
                "concurrency_level": int(prompt_event["concurrency_level"]),
                "workload": workload,
                "heat_load": smoothed_heat,
            }
        )

        previous_heat = smoothed_heat

    return rows


def generate_prompt_trace_dataframe(
    config: CoolSyncConfig,
    scenarios: Sequence[str] | None = None,
    steps_per_scenario: int = 600,
    base_seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate one dataframe by concatenating traces from one or more scenarios.
    """
    if scenarios is None:
        scenarios = DEFAULT_TRACE_SCENARIOS

    all_rows: List[Dict] = []

    for scenario_offset, scenario_name in enumerate(scenarios):
        scenario_seed = None
        if base_seed is not None:
            scenario_seed = base_seed + scenario_offset

        scenario_rows = generate_prompt_trace_rows(
            config=config,
            scenario_name=scenario_name,
            num_steps=steps_per_scenario,
            seed=scenario_seed,
        )
        all_rows.extend(scenario_rows)

    return pd.DataFrame(all_rows)


def build_heat_series_from_dataframe(trace_df: pd.DataFrame) -> List[float]:
    """
    Extract the heat series used by the one-step LSTM.
    """
    if "heat_load" not in trace_df.columns:
        raise ValueError("trace_df must contain a 'heat_load' column")

    return trace_df["heat_load"].astype(float).tolist()


def generate_and_save_trace_dataset(
    config: CoolSyncConfig,
    scenarios: Sequence[str] | None = None,
    steps_per_scenario: int = 600,
    output_path: str = "data/generated/prompt_driven_heat_traces.csv",
) -> Dict:
    """
    Generate the prompt-driven trace dataset and save it for reproducibility.
    """
    trace_df = generate_prompt_trace_dataframe(
        config=config,
        scenarios=scenarios,
        steps_per_scenario=steps_per_scenario,
        base_seed=config.seed,
    )

    ensure_parent_dir(output_path)
    trace_df.to_csv(output_path, index=False)

    return {
        "output_path": output_path,
        "num_rows": int(len(trace_df)),
        "scenarios": list(trace_df["scenario_name"].unique()),
        "columns": list(trace_df.columns),
    }