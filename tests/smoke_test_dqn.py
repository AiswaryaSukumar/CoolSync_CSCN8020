# tests/smoke_test_dqn.py

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
from agents.dqn_agent import DQNAgent


def test_dqn_smoke() -> None:
    """
    Smoke test for DQN components.

    Verifies:
    - environment resets and steps
    - state normalization works
    - DQN agent can select actions, store transitions, and train
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

    state_dim = 6 if preprocessor.use_forecast else 5

    agent = DQNAgent(
        preprocessor=preprocessor,
        state_dim=state_dim,
        num_actions=3,
        learning_rate=1e-3,
        discount_factor=0.99,
        epsilon_start=0.2,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        batch_size=4,
        replay_capacity=100,
        target_update_freq=5,
    )

    state, _ = env.reset()
    norm_state = preprocessor.normalize_state(state)

    terminated = False
    truncated = False
    step_count = 0

    while not (terminated or truncated) and step_count < 8:
        action = agent.select_action(norm_state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        norm_next_state = preprocessor.normalize_state(next_state)

        agent.store_transition(
            state=norm_state,
            action=action,
            reward=reward,
            next_state=norm_next_state,
            done=(terminated or truncated),
        )

        agent.train_step()
        agent.maybe_update_target_network(step_count)

        norm_state = norm_next_state
        step_count += 1

    env.close()
    print("[PASS] smoke_test_dqn")

if __name__ == "__main__":
    test_dqn_smoke()