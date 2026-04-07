# baselines/predictive_threshold_controller.py

from __future__ import annotations


class PredictiveThresholdCoolingController:
    """
    Forecast-aware rule-based baseline controller.

    Uses:
    - current temperature
    - predicted next-step heat

    Action contract:
        0 = decrease cooling
        1 = maintain cooling
        2 = increase cooling

    Expected state contract:
        [
            current_temperature,
            current_workload,
            current_cooling_level,
            ambient_temperature,
            previous_action,
            predicted_heat_next_step,
        ]
    """

    def __init__(
        self,
        safe_temp_min: float = 18.0,
        safe_temp_max: float = 27.0,
        predicted_heat_threshold: float = 0.75,
        low_heat_threshold: float = 0.25,
    ) -> None:
        self.safe_temp_min = safe_temp_min
        self.safe_temp_max = safe_temp_max
        self.predicted_heat_threshold = predicted_heat_threshold
        self.low_heat_threshold = low_heat_threshold

    def select_action(self, state) -> int:
        """
        Select an action using current temperature plus predicted heat.

        Logic:
        - If the system is already too hot, increase cooling immediately.
        - If upcoming heat is predicted to be high, proactively increase cooling.
        - If the system is too cold and upcoming heat is also low, reduce cooling.
        - Otherwise, maintain current cooling.
        """
        current_temperature = float(state[0])
        predicted_heat_next_step = float(state[5])

        # Too hot right now -> increase cooling immediately
        if current_temperature > self.safe_temp_max:
            return 2

        # Forecast indicates near-future heat rise -> proactively cool
        if predicted_heat_next_step > self.predicted_heat_threshold:
            return 2

        # Too cold and no significant future heat expected -> reduce cooling
        if (
            current_temperature < self.safe_temp_min
            and predicted_heat_next_step < self.low_heat_threshold
        ):
            return 0

        # Otherwise keep cooling unchanged
        return 1