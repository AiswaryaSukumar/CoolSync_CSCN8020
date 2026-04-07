# scenarios/scenario_definitions.py

from __future__ import annotations


SCENARIOS = {
    "stable": {
        "description": "Mostly simple and reasoning prompts with low concurrency and low noise.",
        "prompt_type_probabilities": {
            "simple": 0.40,
            "reasoning": 0.25,
            "code": 0.15,
            "long_context": 0.15,
            "burst": 0.05,
        },
        "ambient_mode": "stable",
        "ambient_noise_std": 0.03,
        "thermal_noise_scale": 0.8,
    },
    "sinusoidal": {
        "description": "Balanced traffic with smooth cyclical variation over time.",
        "prompt_type_probabilities": {
            "simple": 0.25,
            "reasoning": 0.20,
            "code": 0.20,
            "long_context": 0.20,
            "burst": 0.15,
        },
        "ambient_mode": "sinusoidal",
        "ambient_noise_std": 0.05,
        "thermal_noise_scale": 1.0,
    },
    "spiky": {
        "description": "Frequent bursts of high-intensity prompts that create sudden heat spikes.",
        "prompt_type_probabilities": {
            "simple": 0.15,
            "reasoning": 0.20,
            "code": 0.20,
            "long_context": 0.15,
            "burst": 0.30,
        },
        "ambient_mode": "stable",
        "ambient_noise_std": 0.06,
        "thermal_noise_scale": 1.1,
    },
    "burst_heavy": {
        "description": "Burst-oriented workload with high concurrency and sustained prompt-induced heat surges.",
        "prompt_type_probabilities": {
            "simple": 0.10,
            "reasoning": 0.15,
            "code": 0.20,
            "long_context": 0.15,
            "burst": 0.40,
        },
        "ambient_mode": "warm_drift",
        "ambient_noise_std": 0.08,
        "thermal_noise_scale": 1.2,
    },
}