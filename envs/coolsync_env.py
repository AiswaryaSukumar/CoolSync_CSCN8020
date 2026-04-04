# envs/coolsync_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Tuple

from configs.default_config import CoolSyncConfig
from envs.env_data_loader   import EnvDataLoader


class CoolSyncEnv(gym.Env):
    """
    CoolSync+ Gymnasium Environment.

    Simulates a single AI data center rack.
    Driven by real workload data from stage4 dataset.
    Optionally uses LSTM forecast as 6th state variable.

    STATE (5D without forecast / 6D with forecast):
      [0] current_temperature   - rack temp in degrees C
      [1] current_workload      - GPU load (0 to 1)
      [2] current_cooling_level - AC level (0 to 10)
      [3] ambient_temperature   - room temp in degrees C
      [4] previous_action       - last action (0, 1, 2)
      [5] predicted_heat        - LSTM forecast (if enabled)

    ACTIONS (3 discrete):
      0 = Decrease cooling
      1 = Maintain cooling
      2 = Increase cooling

    REWARD:
      Penalize: energy, overheating, overcooling, instability
      Reward:   staying in safe temperature zone (18-27 C)
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        config:          Optional[CoolSyncConfig] = None,
        data_path:       str  = 'data/stage4_cooling_control_norm.csv',
        lstm_checkpoint: Optional[str] = None,
        use_forecast:    bool = True,
        use_real_data:   bool = True,
    ):
        super().__init__()

        self.config        = config or CoolSyncConfig()
        self.use_forecast  = use_forecast
        self.use_real_data = use_real_data

        # State dimensions
        self.state_dim = 6 if use_forecast else 5

        # Action space: 3 discrete actions
        self.action_space = spaces.Discrete(3)

        # Observation space
        if use_forecast:
            low  = np.array(
                [0., 0., 0., 0., 0., -5.],
                dtype=np.float32
            )
            high = np.array(
                [50., 1., 10., 50., 2., 5.],
                dtype=np.float32
            )
        else:
            low  = np.array(
                [0., 0., 0., 0., 0.],
                dtype=np.float32
            )
            high = np.array(
                [50., 1., 10., 50., 2.],
                dtype=np.float32
            )

        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32
        )

        # Load real data
        if use_real_data:
            self.data_loader = EnvDataLoader(data_path)
        else:
            self.data_loader = None

        # Load LSTM predictor if requested
        self.predictor = None
        if use_forecast and lstm_checkpoint:
            from forecasting.predict import HeatPredictor
            self.predictor = HeatPredictor(lstm_checkpoint)

        # Internal state variables
        self.current_step    = 0
        self.temperature     = self.config.initial_temperature
        self.workload        = self.config.initial_workload
        self.cooling_level   = self.config.initial_cooling_level
        self.ambient_temp    = self.config.initial_ambient_temp
        self.previous_action = 1
        self.predicted_heat  = 0.0
        self.history         = []
        self._feature_buffer = []

    # ─────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────
    def reset(
        self,
        *,
        seed:    Optional[int]  = None,
        options: Optional[Dict] = None,
    ):
        super().reset(seed=seed)

        self.current_step    = 0
        self.temperature     = float(
            np.random.uniform(22.0, 25.0)
        )
        self.workload        = float(
            np.random.uniform(0.3, 0.7)
        )
        self.cooling_level   = self.config.initial_cooling_level
        self.ambient_temp    = self.config.initial_ambient_temp
        self.previous_action = 1
        self.predicted_heat  = 0.0
        self.history         = []
        self._feature_buffer = []

        return self._get_state(), {}

    # ─────────────────────────────────────────────
    # STEP
    # ─────────────────────────────────────────────
    def step(self, action: int):
        assert self.action_space.contains(action), \
            f"Invalid action: {action}"

        prev_cooling = self.cooling_level

        # 1. Apply action
        self._apply_action(action)

        # 2. Update workload
        self._update_workload()

        # 3. Update ambient temperature
        self._update_ambient()

        # 4. Update rack temperature via physics
        self._update_temperature()

        # 5. Update feature buffer and get forecast
        self._update_feature_buffer()
        if self.use_forecast and self.predictor:
            self.predicted_heat = self._get_forecast()

        # 6. Calculate reward
        reward, info = self._compute_reward(
            action, prev_cooling
        )

        # 7. Update state and history
        self.previous_action = int(action)
        self.current_step   += 1

        self.history.append({
            'step':           self.current_step,
            'temperature':    round(self.temperature, 3),
            'workload':       round(self.workload, 3),
            'cooling_level':  self.cooling_level,
            'action':         int(action),
            'reward':         round(reward, 4),
            'predicted_heat': round(self.predicted_heat, 4),
            'is_overheating': info['is_overheating'],
            'is_overcooling': info['is_overcooling'],
            'energy':         info['energy'],
            'ambient_temp':   round(self.ambient_temp, 3),
        })

        # 8. Check termination
        terminated = (
            self.config.terminate_on_critical
            and self.temperature >= self.config.critical_temp
        )
        truncated = (
            self.current_step >= self.config.episode_length
        )

        return (
            self._get_state(),
            reward,
            terminated,
            truncated,
            info,
        )

    # ─────────────────────────────────────────────
    # RENDER
    # ─────────────────────────────────────────────
    def render(self):
        status = (
            'OVERHEATING' if self.temperature > 27
            else 'OVERCOOLING' if self.temperature < 18
            else 'SAFE'
        )
        print(
            f"Step={self.current_step:3d} | "
            f"Temp={self.temperature:5.2f}C | "
            f"Workload={self.workload:.2f} | "
            f"Cooling={self.cooling_level:2d} | "
            f"Forecast={self.predicted_heat:+.3f} | "
            f"Status={status}"
        )

    # ─────────────────────────────────────────────
    # PRIVATE: get state
    # ─────────────────────────────────────────────
    def _get_state(self) -> np.ndarray:
        state = [
            self.temperature,
            self.workload,
            float(self.cooling_level),
            self.ambient_temp,
            float(self.previous_action),
        ]
        if self.use_forecast:
            state.append(self.predicted_heat)
        return np.array(state, dtype=np.float32)

    # ─────────────────────────────────────────────
    # PRIVATE: apply action
    # ─────────────────────────────────────────────
    def _apply_action(self, action: int):
        if action == 0:
            self.cooling_level -= 1
        elif action == 2:
            self.cooling_level += 1

        self.cooling_level = int(np.clip(
            self.cooling_level,
            self.config.min_cooling_level,
            self.config.max_cooling_level,
        ))

    # ─────────────────────────────────────────────
    # PRIVATE: update workload
    # ─────────────────────────────────────────────
    def _update_workload(self):
        if self.use_real_data and self.data_loader:
            self.workload = self.data_loader.get_workload(
                self.current_step
            )
        else:
            t = self.current_step
            spike = 0.4 if np.random.random() < 0.05 else 0.0
            self.workload = float(np.clip(
                0.5 + 0.3 * np.sin(2 * np.pi * t / 40)
                + spike
                + np.random.normal(0, 0.05),
                0.0, 1.0
            ))

    # ─────────────────────────────────────────────
    # PRIVATE: update ambient temperature
    # ─────────────────────────────────────────────
    def _update_ambient(self):
        offset = 0.0
        if self.use_real_data and self.data_loader:
            offset = self.data_loader.get_ambient_offset(
                self.current_step
            )

        self.ambient_temp = float(np.clip(
            self.ambient_temp
            + np.random.normal(0, 0.05)
            + offset * 0.1,
            self.config.ambient_temp_min,
            self.config.ambient_temp_max,
        ))

    # ─────────────────────────────────────────────
    # PRIVATE: temperature physics equation
    # ─────────────────────────────────────────────
    def _update_temperature(self):
        """
        Core thermal equation:
        T(t+1) = T(t)
               + alpha x workload
               - beta  x cooling_fraction
               + ambient_effect
               + noise
        """
        noise          = np.random.normal(
            0, self.config.noise_std
        )
        cooling_frac   = (
            self.cooling_level /
            self.config.max_cooling_level
        )
        ambient_effect = 0.05 * (
            self.ambient_temp - self.temperature
        )

        self.temperature = float(
            self.temperature
            + self.config.alpha * self.workload
            - self.config.beta  * cooling_frac
            + ambient_effect
            + noise
        )

    # ─────────────────────────────────────────────
    # PRIVATE: build LSTM feature buffer
    # ─────────────────────────────────────────────
    def _update_feature_buffer(self):
        """
        Build feature vector approximating
        the 10 features the LSTM was trained on.

        T_out approximated from current temperature.
        T_celCC approximated from cooling effect.
        Time features calculated from timestep.

        Note: This is an approximation.
        Real deployment would use actual sensors.
        """
        cfg          = self.config
        temp         = self.temperature
        cool_effect  = (
            self.cooling_level / cfg.max_cooling_level
        ) * 2.0

        # Normalize temperature to z-score scale
        # Training data had mean~0, std~1
        # Environment temp range: 18-35C, center ~24C
        temp_z = (temp - 24.0) / 3.0

        # Approximate outlet sensor readings
        t_out_0 = temp_z + 0.20
        t_out_1 = temp_z + 0.15
        t_out_2 = temp_z + 0.10
        t_out_3 = temp_z + 0.05

        # Approximate cooling cell readings
        t_cel_0 = temp_z - cool_effect
        t_cel_1 = temp_z - cool_effect - 0.05
        t_cel_2 = temp_z - cool_effect - 0.10
        t_cel_3 = temp_z - cool_effect - 0.15

        # Time features
        steps_per_day = (24 * 60) / cfg.time_delta_minutes
        frac_of_day   = (
            self.current_step % steps_per_day
        ) / steps_per_day

        hour_sin = float(np.sin(2 * np.pi * frac_of_day))
        hour_cos = float(np.cos(2 * np.pi * frac_of_day))

        features = [
            t_out_0, t_out_1, t_out_2, t_out_3,
            t_cel_0, t_cel_1, t_cel_2, t_cel_3,
            hour_sin, hour_cos,
        ]

        self._feature_buffer.append(features)

        # Keep only last 8 steps
        if len(self._feature_buffer) > 8:
            self._feature_buffer.pop(0)

    # ─────────────────────────────────────────────
    # PRIVATE: get LSTM forecast
    # ─────────────────────────────────────────────
    def _get_forecast(self) -> float:
        """
        Ask LSTM to predict next thermal load.
        Returns 0.0 if not enough history yet.
        """
        if len(self._feature_buffer) < 8:
            return 0.0

        history = np.array(
            self._feature_buffer,
            dtype=np.float32
        )
        return self.predictor.predict(history)

    # ─────────────────────────────────────────────
    # PRIVATE: compute reward
    # ─────────────────────────────────────────────
    def _compute_reward(
        self,
        action:      int,
        prev_cooling: int,
    ) -> Tuple[float, Dict]:
        """
        Reward function priority order:
          1st: Overheating penalty  (w=10.0) - safety
          2nd: Safe zone bonus      (w=5.0)  - ideal target
          3rd: Overcooling penalty  (w=2.0)  - waste
          4th: Instability penalty  (w=1.0)  - wear
          5th: Energy penalty       (w=0.2)  - efficiency
        """
        cfg = self.config

        overheat    = max(
            0.0, self.temperature - cfg.safe_temp_max
        )
        overcool    = max(
            0.0, cfg.safe_temp_min - self.temperature
        )
        instability = abs(self.cooling_level - prev_cooling)
        energy      = float(self.cooling_level)

        in_safe    = (
            cfg.safe_temp_min
            <= self.temperature
            <= cfg.safe_temp_max
        )
        safe_bonus = cfg.w_safe_bonus if in_safe else 0.0

        reward = (
            - cfg.w_energy      * energy
            - cfg.w_overheat    * (overheat ** 2)
            - cfg.w_overcool    * (overcool ** 2)
            - cfg.w_instability * instability
            + safe_bonus
        )

        info = {
            'temperature':     self.temperature,
            'overheat_amount': overheat,
            'overcool_amount': overcool,
            'energy':          energy,
            'instability':     instability,
            'safe_bonus':      safe_bonus,
            'is_overheating':  int(overheat > 0),
            'is_overcooling':  int(overcool > 0),
            'cooling_level':   self.cooling_level,
            'predicted_heat':  self.predicted_heat,
        }

        return float(reward), info