# agents/q_learning_agent.py

from __future__ import annotations

import os
import pickle
import random
from collections import defaultdict
from typing import DefaultDict, Dict, Tuple

import numpy as np

from training.state_preprocessor import StatePreprocessor


class QLearningAgent:
    """
    Tabular Q-learning agent for CoolSync.

    This agent:
    - discretizes continuous environment states using StatePreprocessor
    - selects actions with epsilon-greedy exploration
    - updates Q-values using the standard Bellman Q-learning rule

    Q-learning update:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
    """

    def __init__(
        self,
        preprocessor: StatePreprocessor,
        num_actions: int = 3,
        learning_rate: float = 0.10,
        discount_factor: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ) -> None:
        # Store preprocessing object used to discretize continuous states
        self.preprocessor = preprocessor

        # Number of available discrete actions in the environment
        self.num_actions = num_actions

        # Core Q-learning hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Exploration settings for epsilon-greedy action selection
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Validate hyperparameters early to avoid silent training issues
        self._validate_hyperparameters()

        # Q-table maps a discretized state tuple to an array of action values
        self.q_table: DefaultDict[Tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(self.num_actions, dtype=np.float32)
        )

    def _validate_hyperparameters(self) -> None:
        """
        Validate initialization parameters.
        """
        if self.num_actions <= 0:
            raise ValueError(f"num_actions must be > 0, got {self.num_actions}")

        if not (0.0 < self.learning_rate <= 1.0):
            raise ValueError(
                f"learning_rate must be in (0, 1], got {self.learning_rate}"
            )

        if not (0.0 <= self.discount_factor <= 1.0):
            raise ValueError(
                f"discount_factor must be in [0, 1], got {self.discount_factor}"
            )

        if not (0.0 <= self.epsilon <= 1.0):
            raise ValueError(f"epsilon_start must be in [0, 1], got {self.epsilon}")

        if not (0.0 <= self.epsilon_min <= 1.0):
            raise ValueError(
                f"epsilon_min must be in [0, 1], got {self.epsilon_min}"
            )

        if self.epsilon_min > self.epsilon:
            raise ValueError(
                f"epsilon_min ({self.epsilon_min}) cannot be greater than "
                f"epsilon_start ({self.epsilon})"
            )

        if not (0.0 < self.epsilon_decay <= 1.0):
            raise ValueError(
                f"epsilon_decay must be in (0, 1], got {self.epsilon_decay}"
            )

    def _discretize_state(self, state: np.ndarray) -> Tuple[int, ...]:
        """
        Convert a continuous state vector into a discrete state tuple.
        """
        if state is None:
            raise ValueError("state cannot be None")

        return self.preprocessor.discretize_state(state)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Return the Q-values for a given continuous state.
        """
        discrete_state = self._discretize_state(state)
        return self.q_table[discrete_state]

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy exploration.

        Args:
            state: Continuous environment state
            training: If True, use epsilon-greedy exploration.
                      If False, always exploit the best action.

        Returns:
            Selected discrete action index
        """
        discrete_state = self._discretize_state(state)

        # Explore with probability epsilon during training
        if training and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        # Exploit the action with the highest current Q-value
        return int(np.argmax(self.q_table[discrete_state]))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Apply the Q-learning Bellman update rule.

        Args:
            state: Current continuous state
            action: Action taken in current state
            reward: Immediate reward received
            next_state: Next continuous state
            done: Whether the episode terminated/truncated
        """
        if not (0 <= action < self.num_actions):
            raise ValueError(
                f"action must be in [0, {self.num_actions - 1}], got {action}"
            )

        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)

        # Current estimate of Q(s,a)
        current_q_value = float(self.q_table[discrete_state][action])

        # Bootstrap from next state only if episode is not done
        next_max_q_value = (
            0.0 if done else float(np.max(self.q_table[discrete_next_state]))
        )

        # Bellman target: immediate reward + discounted best future value
        td_target = float(reward) + self.discount_factor * next_max_q_value

        # Temporal-difference error
        td_error = td_target - current_q_value

        # Standard Q-learning update
        self.q_table[discrete_state][action] = (
            current_q_value + self.learning_rate * td_error
        )

    def decay_epsilon(self) -> None:
        """
        Decay epsilon after each episode while respecting epsilon_min.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def set_eval_mode(self) -> None:
        """
        Disable exploration for evaluation-only runs.
        """
        self.epsilon = 0.0

    def set_training_mode(self, epsilon: float | None = None) -> None:
        """
        Re-enable training mode.

        Args:
            epsilon: Optional exploration rate override.
        """
        if epsilon is not None:
            if not (0.0 <= epsilon <= 1.0):
                raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
            self.epsilon = epsilon
        else:
            # If epsilon was set to 0 in eval mode, restore at least epsilon_min
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def reset_q_table(self) -> None:
        """
        Clear all learned Q-values.
        """
        self.q_table = defaultdict(
            lambda: np.zeros(self.num_actions, dtype=np.float32)
        )

    def get_q_table_size(self) -> int:
        """
        Return the number of visited states stored in the Q-table.
        """
        return len(self.q_table)

    def get_best_action(self, state: np.ndarray) -> int:
        """
        Return the greedy action for a state without exploration.
        """
        discrete_state = self._discretize_state(state)
        return int(np.argmax(self.q_table[discrete_state]))

    def save_q_table(self, filepath: str) -> None:
        """
        Save Q-table and metadata to disk.

        Args:
            filepath: Output pickle file path
        """
        directory = os.path.dirname(filepath)

        # Create parent directory only if a directory component exists
        if directory:
            os.makedirs(directory, exist_ok=True)

        payload = {
            "q_table": dict(self.q_table),  # convert defaultdict to regular dict
            "num_actions": self.num_actions,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }

        with open(filepath, "wb") as file:
            pickle.dump(payload, file)

    def load_q_table(self, filepath: str) -> None:
        """
        Load a saved Q-table and metadata from disk.

        Args:
            filepath: Input pickle file path
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Q-table file not found: {filepath}")

        with open(filepath, "rb") as file:
            payload = pickle.load(file)

        loaded_table = payload["q_table"]

        # Restore defaultdict behavior so unseen states default to zero Q-values
        self.q_table = defaultdict(
            lambda: np.zeros(self.num_actions, dtype=np.float32),
            loaded_table,
        )

        self.num_actions = int(payload["num_actions"])
        self.learning_rate = float(payload["learning_rate"])
        self.discount_factor = float(payload["discount_factor"])
        self.epsilon = float(payload["epsilon"])
        self.epsilon_min = float(payload["epsilon_min"])
        self.epsilon_decay = float(payload["epsilon_decay"])

        # Re-validate in case a corrupted or invalid checkpoint was loaded
        self._validate_hyperparameters()

    def summary(self) -> Dict[str, float | int]:
        """
        Return a lightweight summary of the current agent state.
        """
        return {
            "num_actions": self.num_actions,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "q_table_size": self.get_q_table_size(),
        }