# configs/default_config.py

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class CoolSyncConfig:
    # ------------------------------------------------------------------
    # Core simulation settings
    # ------------------------------------------------------------------
    episode_length: int = 200                    # Number of steps per episode
    time_delta_minutes: int = 5                  # Duration of one simulation step
    seed: int = 42                               # Global seed for reproducibility

    # ------------------------------------------------------------------
    # Temperature safety limits
    # ------------------------------------------------------------------
    safe_temp_min: float = 18.0                  # Lower safe operating temperature
    safe_temp_max: float = 27.0                  # Upper safe operating temperature
    critical_temp: float = 35.0                  # Critical overheat threshold

    # ------------------------------------------------------------------
    # Cooling system settings
    # ------------------------------------------------------------------
    min_cooling_level: int = 0                   # Minimum discrete cooling level
    max_cooling_level: int = 10                  # Maximum discrete cooling level
    initial_cooling_level: int = 5               # Initial cooling level at reset
    terminate_on_critical: bool = True           # End episode when critical temperature is reached

    # ------------------------------------------------------------------
    # Initial environmental conditions
    # ------------------------------------------------------------------
    initial_temperature: float = 24.0            # Initial rack temperature
    initial_workload: float = 0.50               # Kept for backward compatibility
    initial_ambient_temp: float = 22.0           # Initial room/ambient temperature
    ambient_temp_min: float = 20.0               # Minimum ambient temperature
    ambient_temp_max: float = 27.0               # Maximum ambient temperature

    # ------------------------------------------------------------------
    # Thermal dynamics
    # Old names are kept for compatibility, new names are the target.
    # ------------------------------------------------------------------
    alpha: float = 1.8                           # Backward-compatible old field
    beta: float = 1.2                            # Backward-compatible old field
    noise_std: float = 0.15                      # Backward-compatible old field

    alpha_heat: float = 1.8                      # Heat-to-temperature gain
    beta_cooling: float = 1.2                    # Cooling effectiveness coefficient
    ambient_coupling: float = 0.05               # Effect of ambient temperature on rack temperature
    thermal_noise_std: float = 0.15              # Random thermal noise

    # ------------------------------------------------------------------
    # Reward weights
    # ------------------------------------------------------------------
    w_energy: float = 0.20                       # Energy penalty weight
    w_overheat: float = 10.0                     # Overheating penalty weight
    w_overcool: float = 2.0                      # Overcooling penalty weight
    w_instability: float = 1.0                   # Cooling change penalty weight
    w_safe_bonus: float = 5.0                    # Bonus for staying in safe range

    # ------------------------------------------------------------------
    # Prompt generation settings
    # ------------------------------------------------------------------
    prompt_types: List[str] = field(default_factory=lambda: [
        "simple",
        "reasoning",
        "code",
        "long_context",
        "burst",
    ])

    # Default prompt distribution for a balanced generic run
    prompt_type_probabilities: Dict[str, float] = field(default_factory=lambda: {
        "simple": 0.25,
        "reasoning": 0.20,
        "code": 0.20,
        "long_context": 0.20,
        "burst": 0.15,
    })

    # Prompt length ranges
    prompt_length_ranges: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "simple": (50, 300),
        "reasoning": (200, 800),
        "code": (300, 1200),
        "long_context": (1000, 4000),
        "burst": (150, 1000),
    })

    # Complexity score ranges in [0, 1]
    complexity_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "simple": (0.10, 0.30),
        "reasoning": (0.50, 0.80),
        "code": (0.60, 0.95),
        "long_context": (0.70, 1.00),
        "burst": (0.40, 0.85),
    })

    # Concurrency level ranges
    concurrency_ranges: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "simple": (1, 2),
        "reasoning": (1, 3),
        "code": (1, 3),
        "long_context": (1, 2),
        "burst": (3, 8),
    })

    # ------------------------------------------------------------------
    # Prompt-to-heat mapping settings
    # ------------------------------------------------------------------
    max_prompt_length_for_norm: int = 4000       # Used to normalize prompt length
    max_concurrency_for_norm: int = 8            # Used to normalize concurrency

    heat_length_weight: float = 0.35             # Weight for normalized prompt length
    heat_complexity_weight: float = 0.40         # Weight for complexity score
    heat_concurrency_weight: float = 0.25        # Weight for normalized concurrency
    heat_noise_std: float = 0.03                 # Noise added to generated heat

    min_heat_load: float = 0.0                   # Minimum generated heat
    max_heat_load: float = 1.2                   # Maximum generated heat
    min_workload: float = 0.0                    # Minimum workload
    max_workload: float = 1.0                    # Maximum workload
    workload_from_heat_scale: float = 0.95       # Workload derived from heat

    # ------------------------------------------------------------------
    # Nonlinear cooling energy model settings
    # ------------------------------------------------------------------
    fan_power_coeff: float = 0.30                # Fan power coefficient
    fan_power_exponent: float = 2.0              # Nonlinear fan curve exponent

    compressor_power_coeff: float = 2.50         # Compressor contribution coefficient
    base_cop: float = 3.5                        # Base coefficient of performance
    cop_reference_temp: float = 22.0             # Reference ambient temperature
    cop_temp_sensitivity: float = 0.08           # COP degradation with higher ambient temp
    min_cop: float = 1.5                         # Lower COP bound

    # ------------------------------------------------------------------
    # Forecasting settings
    # ------------------------------------------------------------------
    lstm_lookback: int = 8                       # Kept for backward compatibility
    lstm_hidden: int = 64                        # Old/compat field
    lstm_layers: int = 2                         # Old/compat field
    lstm_dropout: float = 0.2                    # Old/compat field
    lstm_input_size: int = 10                    # Old/compat field

    forecast_sequence_length: int = 10           # Sequence length used by new forecast bridge
    forecast_horizon: int = 1                    # One-step forecast only

    # ------------------------------------------------------------------
    # Normalization settings
    # ------------------------------------------------------------------
    temp_min_for_norm: float = 0.0
    temp_max_for_norm: float = 50.0
    workload_min_for_norm: float = 0.0
    workload_max_for_norm: float = 1.0
    heat_min_for_norm: float = 0.0
    heat_max_for_norm: float = 1.2
    action_min_for_norm: int = 0
    action_max_for_norm: int = 2