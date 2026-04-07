# tests/smoke_test_q_learning.py

from __future__ import annotations

import os
import sys

# Add project root to Python path so package-style imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.default_config import CoolSyncConfig
from envs.coolsync_env import CoolSyncEnv
from training.state_preprocessor import StatePreprocessor
from agents.q_learning_agent import QLearningAgent


def test_q_learning_smoke() -> None:
    """
    Smoke test for Q-learning components.

    Verifies:
    - environment resets and steps
    - state preprocessing works
    - Q-learning agent selects and updates actions
    """
    config = CoolSyncConfig()

    env = CoolSyncEnv(
        config=config,
        scenario_name="stable",
        use_forecast=True,
        forecast_model=None,
        forecast_device=None,
    )

    preprocessor = StatePreprocessor(config=config, use_forecast=True)

    agent = QLearningAgent(
        preprocessor=preprocessor,
        num_actions=3,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon_start=0.2,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    )

    state, _ = env.reset()
    discrete_state = preprocessor.discretize_state(state)

    terminated = False
    truncated = False
    step_count = 0

    while not (terminated or truncated) and step_count < 5:
        action = agent.select_action(discrete_state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        discrete_next_state = preprocessor.discretize_state(next_state)

        agent.update(
            state=discrete_state,
            action=action,
            reward=reward,
            next_state=discrete_next_state,
            done=(terminated or truncated),
        )

        discrete_state = discrete_next_state
        step_count += 1

    q_table_size = agent.get_q_table_size()
    assert q_table_size > 0, "Q-table should contain at least one visited state"

    env.close()
    print("[PASS] smoke_test_q_learning")


if __name__ == "__main__":
    test_q_learning_smoke()