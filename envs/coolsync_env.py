# envs/coolsync_env.py

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from configs.default_config import CoolSyncConfig
from forecasting.forecast_utils import predict_next_heat_with_fallback
from prompt_model.prompt_generator import generate_prompt_features
from prompt_model.prompt_to_heat import prompt_to_workload_and_heat
from scenarios.scenario_definitions import SCENARIOS
from utils.energy_model import compute_total_cooling_energy
from utils.seed import set_global_seed


class CoolSyncEnv(gym.Env):
    """
    Prompt-aware predictive cooling environment.

    Final state contract:
        [
            current_temperature,
            current_workload,
            current_cooling_level,
            ambient_temperature,
            previous_action,
            predicted_heat_next_step,
        ]

    Action contract:
        0 = decrease cooling
        1 = maintain cooling
        2 = increase cooling
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: Optional[CoolSyncConfig] = None,
        scenario_name: str = "stable",
        use_forecast: bool = True,
        forecast_model=None,
        forecast_device=None,
    ) -> None:
        super().__init__()

        # Use caller-provided configuration or default project config
        self.config = config if config is not None else CoolSyncConfig()

        # Set reproducible seed
        set_global_seed(self.config.seed)

        # Validate requested scenario
        if scenario_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario_name: {scenario_name}")

        self.scenario_name = scenario_name
        self.scenario_config = SCENARIOS[scenario_name]

        # Forecast behavior controls whether predicted heat is visible in the state
        self.use_forecast = use_forecast
        self.forecast_model = forecast_model
        self.forecast_device = forecast_device

        # Action space:
        # 0 = decrease cooling
        # 1 = maintain
        # 2 = increase cooling
        self.action_space = spaces.Discrete(3)

        # Observation space follows the locked 6D contract
        self.observation_space = spaces.Box(
            low=np.array(
                [
                    0.0,                                   # current_temperature
                    0.0,                                   # current_workload
                    self.config.min_cooling_level,         # current_cooling_level
                    0.0,                                   # ambient_temperature
                    0.0,                                   # previous_action
                    self.config.min_heat_load,             # predicted_heat_next_step
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    50.0,                                  # current_temperature
                    1.0,                                   # current_workload
                    self.config.max_cooling_level,         # current_cooling_level
                    50.0,                                  # ambient_temperature
                    2.0,                                   # previous_action
                    self.config.max_heat_load,             # predicted_heat_next_step
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # Internal episode state
        self.current_step: int = 0
        self.temperature: float = self.config.initial_temperature
        self.ambient_temperature: float = self.config.initial_ambient_temp
        self.cooling_level: int = self.config.initial_cooling_level
        self.previous_action: int = 1
        self.current_workload: float = 0.0
        self.current_heat_load: float = 0.0
        self.predicted_heat_next_step: float = 0.0

        # Rolling history used by forecast bridge
        self.heat_history: Deque[float] = deque(
            maxlen=self.config.forecast_sequence_length
        )

        # Per-step episode history for metrics and plots
        self.history = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state(self) -> np.ndarray:
        """
        Return the current state in the locked order.
        """
        forecast_value = self.predicted_heat_next_step if self.use_forecast else 0.0

        return np.array(
            [
                self.temperature,
                self.current_workload,
                self.cooling_level,
                self.ambient_temperature,
                self.previous_action,
                forecast_value,
            ],
            dtype=np.float32,
        )

    def _sample_prompt_event(self) -> Dict:
        """
        Generate one structured prompt event and convert it to workload + heat.
        """
        prompt_features = generate_prompt_features(
            config=self.config,
            prompt_type_probabilities=self.scenario_config["prompt_type_probabilities"],
        )

        prompt_event = prompt_to_workload_and_heat(
            prompt_features=prompt_features,
            config=self.config,
        )

        return prompt_event

    def _update_ambient_temperature(self) -> None:
        """
        Update ambient temperature according to the active scenario.
        """
        ambient_mode = self.scenario_config["ambient_mode"]
        ambient_noise_std = self.scenario_config["ambient_noise_std"]

        if ambient_mode == "stable":
            self.ambient_temperature += np.random.normal(0.0, ambient_noise_std)

        elif ambient_mode == "sinusoidal":
            sinusoidal_shift = 0.15 * np.sin(2 * np.pi * self.current_step / 40.0)
            self.ambient_temperature += (
                sinusoidal_shift + np.random.normal(0.0, ambient_noise_std)
            )

        elif ambient_mode == "warm_drift":
            self.ambient_temperature += (
                0.03 + np.random.normal(0.0, ambient_noise_std)
            )

        else:
            self.ambient_temperature += np.random.normal(0.0, ambient_noise_std)

        self.ambient_temperature = float(
            np.clip(
                self.ambient_temperature,
                self.config.ambient_temp_min,
                self.config.ambient_temp_max,
            )
        )

    def _predict_next_heat(self) -> float:
        """
        Predict next-step heat using rolling recent heat history.

        If a trained model is unavailable, use a safe fallback.
        """
        recent_heat_history = list(self.heat_history)

        predicted_heat = predict_next_heat_with_fallback(
            recent_heat_history=recent_heat_history,
            sequence_length=self.config.forecast_sequence_length,
            forecast_model=self.forecast_model,
            device=self.forecast_device,
            fallback_value=self.current_heat_load,
        )

        return float(
            np.clip(
                predicted_heat,
                self.config.min_heat_load,
                self.config.max_heat_load,
            )
        )

    def _apply_action(self, action: int) -> None:
        """
        Apply delta-action cooling control.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        if action == 0:
            self.cooling_level -= 1
        elif action == 2:
            self.cooling_level += 1

        self.cooling_level = int(
            np.clip(
                self.cooling_level,
                self.config.min_cooling_level,
                self.config.max_cooling_level,
            )
        )

    def _update_temperature(self) -> None:
        """
        Update rack temperature using the thermal transition equation.

        T(t+1) = T(t)
                 + alpha_heat * current_heat_load
                 - beta_cooling * cooling_effect
                 + ambient_coupling * (ambient_temperature - T)
                 + noise
        """
        cooling_effect = self.cooling_level / max(1, self.config.max_cooling_level)

        ambient_effect = self.config.ambient_coupling * (
            self.ambient_temperature - self.temperature
        )

        thermal_noise_scale = self.scenario_config["thermal_noise_scale"]
        thermal_noise = np.random.normal(
            0.0,
            self.config.thermal_noise_std * thermal_noise_scale,
        )

        self.temperature = float(
            self.temperature
            + self.config.alpha_heat * self.current_heat_load
            - self.config.beta_cooling * cooling_effect
            + ambient_effect
            + thermal_noise
        )

    def _compute_reward(self, previous_cooling_level: int) -> tuple[float, Dict]:
        """
        Compute reward using:
        - cooling energy usage penalty
        - overheating penalty
        - overcooling penalty
        - instability penalty
        - safe-zone bonus
        """
        energy = compute_total_cooling_energy(
            cooling_level=self.cooling_level,
            ambient_temperature=self.ambient_temperature,
            config=self.config,
        )

        overheat_amount = max(0.0, self.temperature - self.config.safe_temp_max)
        overcool_amount = max(0.0, self.config.safe_temp_min - self.temperature)
        instability = abs(self.cooling_level - previous_cooling_level)

        safe_zone_bonus = (
            1.0
            if self.config.safe_temp_min <= self.temperature <= self.config.safe_temp_max
            else 0.0
        )

        reward = (
            - self.config.w_energy * energy
            - self.config.w_overheat * (overheat_amount ** 2)
            - self.config.w_overcool * (overcool_amount ** 2)
            - self.config.w_instability * instability
            + self.config.w_safe_bonus * safe_zone_bonus
        )

        info = {
            "energy": float(energy),
            "temperature": float(self.temperature),
            "ambient_temperature": float(self.ambient_temperature),
            "workload": float(self.current_workload),
            "heat_load": float(self.current_heat_load),
            "predicted_heat_next_step": float(self.predicted_heat_next_step),
            "cooling_level": int(self.cooling_level),
            "overheat_amount": float(overheat_amount),
            "overcool_amount": float(overcool_amount),
            "instability": float(instability),
            "safe_zone_bonus": float(safe_zone_bonus),
            "is_overheating": int(overheat_amount > 0.0),
            "is_overcooling": int(overcool_amount > 0.0),
        }

        return float(reward), info

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed)

        if seed is not None:
            set_global_seed(seed)

        self.current_step = 0
        self.temperature = self.config.initial_temperature
        self.ambient_temperature = self.config.initial_ambient_temp
        self.cooling_level = self.config.initial_cooling_level
        self.previous_action = 1
        self.current_workload = 0.0
        self.current_heat_load = 0.0
        self.predicted_heat_next_step = 0.0
        self.history = []

        # Initialize heat history with zeros so forecast window is valid
        self.heat_history = deque(
            [0.0] * self.config.forecast_sequence_length,
            maxlen=self.config.forecast_sequence_length,
        )

        return self._get_state(), {}

    def step(self, action: int):
        """
        Execute one environment step.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        previous_cooling_level = self.cooling_level

        # 1. Apply action
        self._apply_action(action)

        # 2. Advance simulation clock
        self.current_step += 1

        # 3. Update ambient conditions
        self._update_ambient_temperature()

        # 4. Generate prompt-driven workload and heat
        prompt_event = self._sample_prompt_event()
        self.current_workload = float(prompt_event["workload"])
        self.current_heat_load = float(prompt_event["heat_load"])

        # 5. Update heat history and forecast next-step heat
        self.heat_history.append(self.current_heat_load)
        self.predicted_heat_next_step = self._predict_next_heat()

        # 6. Update thermal state
        self._update_temperature()

        # 7. Compute reward and diagnostics
        reward, info = self._compute_reward(
            previous_cooling_level=previous_cooling_level
        )

        # 8. Track action
        self.previous_action = int(action)

        # 9. Stopping conditions
        terminated = (
            self.config.terminate_on_critical
            and self.temperature >= self.config.critical_temp
        )
        truncated = self.current_step >= self.config.episode_length

        # 10. Store history for metrics / plots / comparisons
        self.history.append(
            {
                "step": int(self.current_step),
                "action": int(action),
                "reward": float(reward),
                "temperature": float(self.temperature),
                "ambient_temperature": float(self.ambient_temperature),
                "workload": float(self.current_workload),
                "heat_load": float(self.current_heat_load),
                "predicted_heat_next_step": float(self.predicted_heat_next_step),
                "cooling_level": int(self.cooling_level),
                "energy": float(info["energy"]),
                "is_overheating": int(info["is_overheating"]),
                "is_overcooling": int(info["is_overcooling"]),
            }
        )

        return self._get_state(), reward, terminated, truncated, info

    def render(self):
        """
        Lightweight text render for debugging.
        """
        print(
            f"Step={self.current_step} | "
            f"Temp={self.temperature:.2f}°C | "
            f"Workload={self.current_workload:.2f} | "
            f"Heat={self.current_heat_load:.2f} | "
            f"PredHeat={self.predicted_heat_next_step:.2f} | "
            f"Cooling={self.cooling_level} | "
            f"Ambient={self.ambient_temperature:.2f}°C"
        )

    def close(self):
        """
        Placeholder for environment cleanup.
        """
        pass