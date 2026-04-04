# baselines/static_controller.py

class StaticCoolingController:
    """
    Always uses the same cooling action regardless
    of temperature or workload.

    This is the simplest possible baseline.
    No intelligence, no adaptation.

    fixed_action:
      0 = always decrease cooling
      1 = always maintain (default)
      2 = always increase cooling
    """

    def __init__(self, fixed_action: int = 1):
        self.fixed_action = fixed_action

    def select_action(self, state, training: bool = False):
        return self.fixed_action