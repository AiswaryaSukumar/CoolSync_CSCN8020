# prompt_model/prompt_generator.py

from __future__ import annotations

import random
from typing import Dict, Optional

import numpy as np

from configs.default_config import CoolSyncConfig
from prompt_model.prompt_features import PROMPT_TYPES, PromptFeatures


def validate_prompt_probabilities(probabilities: Dict[str, float]) -> None:
    """
    Validate that all expected prompt types exist and that probabilities sum to 1.
    """
    missing_types = [prompt_type for prompt_type in PROMPT_TYPES if prompt_type not in probabilities]
    if missing_types:
        raise ValueError(f"Missing probabilities for prompt types: {missing_types}")

    total_probability = sum(probabilities.values())
    if not np.isclose(total_probability, 1.0, atol=1e-6):
        raise ValueError(
            f"Prompt probabilities must sum to 1.0, but got {total_probability:.6f}"
        )


def sample_prompt_type(
    config: CoolSyncConfig,
    prompt_type_probabilities: Optional[Dict[str, float]] = None,
) -> str:
    """
    Sample one prompt type using either scenario-specific probabilities
    or the default probabilities from the config.
    """
    probabilities = (
        prompt_type_probabilities
        if prompt_type_probabilities is not None
        else config.prompt_type_probabilities
    )

    validate_prompt_probabilities(probabilities)

    prompt_types = list(probabilities.keys())
    weights = list(probabilities.values())

    # Randomly choose one prompt category according to the provided weights.
    return random.choices(prompt_types, weights=weights, k=1)[0]


def sample_prompt_length(prompt_type: str, config: CoolSyncConfig) -> int:
    """
    Sample prompt length from the configured range for the given prompt type.
    """
    low, high = config.prompt_length_ranges[prompt_type]
    return random.randint(low, high)


def sample_complexity_score(prompt_type: str, config: CoolSyncConfig) -> float:
    """
    Sample a normalized complexity score in [0, 1].
    """
    low, high = config.complexity_ranges[prompt_type]
    return float(np.random.uniform(low, high))


def sample_concurrency_level(prompt_type: str, config: CoolSyncConfig) -> int:
    """
    Sample concurrency for the prompt type.
    Burst prompts should naturally trend higher because their configured
    range should be larger inside default_config.py.
    """
    low, high = config.concurrency_ranges[prompt_type]
    return random.randint(low, high)


def generate_prompt_features(
    config: CoolSyncConfig,
    prompt_type_probabilities: Optional[Dict[str, float]] = None,
) -> PromptFeatures:
    """
    Main entry point for generating one structured prompt event.
    """
    prompt_type = sample_prompt_type(
        config=config,
        prompt_type_probabilities=prompt_type_probabilities,
    )

    prompt_length = sample_prompt_length(prompt_type=prompt_type, config=config)
    complexity_score = sample_complexity_score(prompt_type=prompt_type, config=config)
    concurrency_level = sample_concurrency_level(prompt_type=prompt_type, config=config)

    return PromptFeatures(
        prompt_type=prompt_type,
        prompt_length=prompt_length,
        complexity_score=complexity_score,
        concurrency_level=concurrency_level,
    )