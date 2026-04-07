# utils/metrics.py

from __future__ import annotations

from typing import Dict, List

import numpy as np


def compute_recovery_time_after_spike(
    episode_history: List[Dict],
    safe_temp_max: float = 27.0,
) -> float:
    """
    Estimate recovery time after the first overheating spike.

    Recovery time is the number of steps from the first overheating event
    until the temperature returns to or below the safe upper bound.
    """
    if not episode_history:
        return 0.0

    spike_index = None
    recovery_index = None

    for idx, row in enumerate(episode_history):
        if row["temperature"] > safe_temp_max:
            spike_index = idx
            break

    if spike_index is None:
        return 0.0

    for idx in range(spike_index + 1, len(episode_history)):
        if episode_history[idx]["temperature"] <= safe_temp_max:
            recovery_index = idx
            break

    if recovery_index is None:
        return float(len(episode_history) - spike_index)

    return float(recovery_index - spike_index)


def summarize_episode(
    episode_history: List[Dict],
    safe_temp_max: float = 27.0,
) -> Dict:
    """
    Summarize one episode into evaluation-ready metrics.
    """
    if not episode_history:
        return {
            "total_reward": 0.0,
            "total_energy": 0.0,
            "avg_temperature": 0.0,
            "max_temperature": 0.0,
            "overheat_count": 0,
            "overcool_count": 0,
            "cooling_variance": 0.0,
            "recovery_time": 0.0,
        }

    rewards = [row["reward"] for row in episode_history]
    energies = [row["energy"] for row in episode_history]
    temperatures = [row["temperature"] for row in episode_history]
    cooling_levels = [row["cooling_level"] for row in episode_history]
    overheat_flags = [row["is_overheating"] for row in episode_history]
    overcool_flags = [row["is_overcooling"] for row in episode_history]

    recovery_time = compute_recovery_time_after_spike(
        episode_history=episode_history,
        safe_temp_max=safe_temp_max,
    )

    return {
        "total_reward": float(np.sum(rewards)),
        "total_energy": float(np.sum(energies)),
        "avg_temperature": float(np.mean(temperatures)),
        "max_temperature": float(np.max(temperatures)),
        "overheat_count": int(np.sum(overheat_flags)),
        "overcool_count": int(np.sum(overcool_flags)),
        "cooling_variance": float(np.var(cooling_levels)),
        "recovery_time": float(recovery_time),
    }