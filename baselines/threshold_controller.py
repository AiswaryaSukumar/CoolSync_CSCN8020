# baselines/threshold_controller.py

class ThresholdCoolingController:
    """
    Reacts only after temperature crosses a threshold.

    This is the classic reactive approach.
    It cannot predict future heat.
    It always responds too late.

    This is exactly what CoolSync+ improves upon.
    """

    def __init__(
        self,
        safe_temp_min: float = 18.0,
        safe_temp_max: float = 27.0,
    ):
        self.safe_min = safe_temp_min
        self.safe_max = safe_temp_max

    def select_action(self, state, training: bool = False):
        # state[0] = current temperature
        temperature = state[0]

        if temperature > self.safe_max:
            return 2   # too hot - increase cooling
        elif temperature < self.safe_min:
            return 0   # too cold - decrease cooling
        else:
            return 1   # safe - maintain