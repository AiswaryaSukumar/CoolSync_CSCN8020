# tests/smoke_test_env.py

from __future__ import annotations

import os
import sys

# Add project root to Python path so package-style imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.default_config import CoolSyncConfig
from envs.coolsync_env import CoolSyncEnv


def test_environment_smoke() -> None:
    """
    Basic smoke test for the CoolSync environment.

    Verifies:
    - reset works
    - step works
    - state shape is correct
    - reward is numeric
    - history is populated
    """
    config = CoolSyncConfig()

    env = CoolSyncEnv(
        config=config,
        scenario_name="stable",
        use_forecast=True,
        forecast_model=None,
        forecast_device=None,
    )

    state, info = env.reset()

    assert len(state) == 6, f"Expected state length 6, got {len(state)}"

    terminated = False
    truncated = False
    step_count = 0

    while not (terminated or truncated) and step_count < 5:
        action = 1  # Maintain cooling
        next_state, reward, terminated, truncated, step_info = env.step(action)

        assert len(next_state) == 6, f"Expected next_state length 6, got {len(next_state)}"
        assert isinstance(float(reward), float), "Reward is not numeric"
        assert "temperature" in step_info, "Missing temperature in step info"
        assert "energy" in step_info, "Missing energy in step info"

        step_count += 1

    assert len(env.history) > 0, "Environment history should not be empty after stepping"

    env.close()
    print("[PASS] smoke_test_env")


if __name__ == "__main__":
    test_environment_smoke()