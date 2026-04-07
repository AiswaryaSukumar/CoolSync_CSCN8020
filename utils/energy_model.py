# utils/energy_model.py

from __future__ import annotations

from configs.default_config import CoolSyncConfig


def normalize_cooling_level(cooling_level: int, config: CoolSyncConfig) -> float:
    """
    Normalize cooling level into [0, 1].
    """
    if config.max_cooling_level == config.min_cooling_level:
        return 0.0

    normalized = (
        (cooling_level - config.min_cooling_level)
        / (config.max_cooling_level - config.min_cooling_level)
    )

    return max(0.0, min(1.0, float(normalized)))


def compute_fan_power(cooling_level: int, config: CoolSyncConfig) -> float:
    """
    Compute fan power using a nonlinear relation to cooling level.
    """
    cooling_norm = normalize_cooling_level(cooling_level, config)

    return float(
        config.fan_power_coeff * (cooling_norm ** config.fan_power_exponent)
    )


def compute_cop(ambient_temperature: float, config: CoolSyncConfig) -> float:
    """
    Compute cooling system coefficient of performance (COP).

    Higher ambient temperature typically reduces cooling efficiency.
    """
    ambient_delta = max(0.0, ambient_temperature - config.cop_reference_temp)

    cop = config.base_cop - config.cop_temp_sensitivity * ambient_delta

    return float(max(config.min_cop, cop))


def compute_total_cooling_energy(
    cooling_level: int,
    ambient_temperature: float,
    config: CoolSyncConfig,
) -> float:
    """
    Compute total cooling energy using:
    - nonlinear fan power
    - compressor contribution adjusted by COP
    """
    cooling_norm = normalize_cooling_level(cooling_level, config)

    fan_power = compute_fan_power(cooling_level=cooling_level, config=config)
    cop = compute_cop(ambient_temperature=ambient_temperature, config=config)

    compressor_power = 0.0
    if cop > 0.0:
        compressor_power = config.compressor_power_coeff * cooling_norm / cop

    total_energy = fan_power + compressor_power

    return float(total_energy)