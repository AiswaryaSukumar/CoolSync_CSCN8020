# baselines/pid_controller.py

class SimplePIDCoolingController:
    """
    PID Controller - reacts to temperature error.

    P = proportional to distance from target
    D = proportional to rate of change

    Better than threshold but still purely reactive.
    Cannot predict future heat from workload signals.
    Requires manual tuning of kp and kd parameters.
    """

    def __init__(
        self,
        target_temp: float = 24.0,
        kp:          float = 0.8,
        kd:          float = 0.1,
    ):
        self.target     = target_temp
        self.kp         = kp
        self.kd         = kd
        self.prev_error = 0.0

    def select_action(self, state, training: bool = False):
        current_temp = state[0]

        # Error from target
        error = current_temp - self.target

        # Rate of change of error
        delta = error - self.prev_error

        # Control signal
        signal = self.kp * error + self.kd * delta

        self.prev_error = error

        if signal > 0.5:
            return 2   # increase cooling
        elif signal < -0.5:
            return 0   # decrease cooling
        return 1       # maintain