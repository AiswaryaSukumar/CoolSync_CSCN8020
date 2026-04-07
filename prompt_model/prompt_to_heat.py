# prompt_model/prompt_to_heat.py

from __future__ import annotations

from typing import Dict

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch may not be available in some lightweight environments
    torch = None

from configs.default_config import CoolSyncConfig
from prompt_model.prompt_features import PromptFeatures


def clip_value(value: float, min_value: float, max_value: float) -> float:
    """Clip a floating-point value into a valid range."""
    return float(np.clip(value, min_value, max_value))


def normalize_prompt_length(prompt_length: int, config: CoolSyncConfig) -> float:
    """Normalize prompt length into approximately [0, 1]."""
    return clip_value(
        value=prompt_length / config.max_prompt_length_for_norm,
        min_value=0.0,
        max_value=1.0,
    )


def normalize_concurrency(concurrency_level: int, config: CoolSyncConfig) -> float:
    """Normalize concurrency level into approximately [0, 1]."""
    return clip_value(
        value=concurrency_level / config.max_concurrency_for_norm,
        min_value=0.0,
        max_value=1.0,
    )


def get_compute_device() -> str:
    """Return the compute device used by the prompt simulation layer."""
    if torch is not None and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def simulate_concurrent_request_load(
    prompt_features: PromptFeatures,
    config: CoolSyncConfig,
) -> Dict[str, float]:
    """
    Simulate concurrent AI inference requests using batched tensor math.

    This adds a truthful technical layer for the dashboard:
    - if CUDA is available, the tensor batch runs on GPU
    - otherwise it falls back to CPU

    The goal is not to benchmark a real serving cluster, but to produce a
    realistic compute-backed concurrency signal for the cooling simulator.
    """
    normalized_length = normalize_prompt_length(prompt_features.prompt_length, config)
    normalized_concurrency = normalize_concurrency(prompt_features.concurrency_level, config)

    request_count = max(1, int(prompt_features.concurrency_level))
    feature_dim = 4

    if torch is None:
        simulated_load = (
            0.40 * normalized_length
            + 0.35 * prompt_features.complexity_score
            + 0.25 * normalized_concurrency
        )
        variability = 0.05 * normalized_concurrency
        return {
            'compute_device': 'cpu',
            'simulated_concurrency': float(request_count),
            'batch_workload': clip_value(simulated_load, 0.0, 1.0),
            'batch_variability': clip_value(variability, 0.0, 1.0),
        }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_vector = torch.tensor(
        [
            normalized_length,
            float(prompt_features.complexity_score),
            normalized_concurrency,
            1.0 if prompt_features.prompt_type in {'code', 'burst'} else 0.45,
        ],
        dtype=torch.float32,
        device=device,
    )

    noise = 0.08 * torch.randn(request_count, feature_dim, device=device)
    request_tensor = torch.clamp(base_vector.repeat(request_count, 1) + noise, 0.0, 1.0)

    per_request_load = (
        0.35 * request_tensor[:, 0]
        + 0.35 * request_tensor[:, 1]
        + 0.20 * request_tensor[:, 2]
        + 0.10 * request_tensor[:, 3]
    )

    batch_workload = torch.clamp(per_request_load.mean(), 0.0, 1.0).item()
    batch_variability = torch.clamp(per_request_load.std(unbiased=False), 0.0, 1.0).item()

    return {
        'compute_device': str(device),
        'simulated_concurrency': float(request_count),
        'batch_workload': float(batch_workload),
        'batch_variability': float(batch_variability),
    }


def compute_workload_intensity(
    prompt_features: PromptFeatures,
    config: CoolSyncConfig,
) -> float:
    """
    Convert prompt features into workload intensity.

    The result blends feature-based workload with a batched concurrency
    simulation so the dashboard can credibly claim CUDA-backed simulation.
    """
    normalized_length = normalize_prompt_length(prompt_features.prompt_length, config)
    normalized_concurrency = normalize_concurrency(prompt_features.concurrency_level, config)

    feature_workload = (
        0.35 * normalized_length
        + 0.40 * prompt_features.complexity_score
        + 0.25 * normalized_concurrency
    )

    concurrent_sim = simulate_concurrent_request_load(prompt_features, config)

    workload = 0.60 * feature_workload + 0.40 * concurrent_sim['batch_workload']

    return clip_value(
        value=workload,
        min_value=config.min_workload,
        max_value=config.max_workload,
    )


def compute_heat_load(
    prompt_features: PromptFeatures,
    config: CoolSyncConfig,
    workload_intensity: float | None = None,
) -> float:
    """
    Convert prompt features into a heat load.

    Heat model:
        heat = a1 * workload
             + a2 * complexity_score
             + a3 * normalized_concurrency
             + a4 * batch_variability
             + noise
    """
    if workload_intensity is None:
        workload_intensity = compute_workload_intensity(prompt_features, config)

    normalized_concurrency = normalize_concurrency(prompt_features.concurrency_level, config)
    concurrent_sim = simulate_concurrent_request_load(prompt_features, config)
    batch_variability = concurrent_sim['batch_variability']

    noise = np.random.normal(0.0, config.heat_noise_std)

    heat_load = (
        0.45 * workload_intensity
        + 0.25 * prompt_features.complexity_score
        + 0.20 * normalized_concurrency
        + 0.10 * batch_variability
        + noise
    )

    return clip_value(
        value=heat_load,
        min_value=config.min_heat_load,
        max_value=config.max_heat_load,
    )


def prompt_to_workload_and_heat(
    prompt_features: PromptFeatures,
    config: CoolSyncConfig,
) -> Dict:
    """Main public conversion function used by the environment and API."""
    concurrency_sim = simulate_concurrent_request_load(
        prompt_features=prompt_features,
        config=config,
    )

    workload = compute_workload_intensity(
        prompt_features=prompt_features,
        config=config,
    )

    heat_load = compute_heat_load(
        prompt_features=prompt_features,
        config=config,
        workload_intensity=workload,
    )

    return {
        'prompt_type': prompt_features.prompt_type,
        'prompt_length': prompt_features.prompt_length,
        'complexity_score': float(prompt_features.complexity_score),
        'concurrency_level': int(prompt_features.concurrency_level),
        'workload': float(workload),
        'heat_load': float(heat_load),
        'compute_device': concurrency_sim['compute_device'],
        'simulated_concurrency': float(concurrency_sim['simulated_concurrency']),
        'batch_variability': float(concurrency_sim['batch_variability']),
    }
