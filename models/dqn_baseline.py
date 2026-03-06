"""
DQN Baseline: standard Deep Q-Network for Nim via self-play.

Uses a fixed-size board representation. The Q-network estimates action values
for each possible (heap_index, remove_count) pair. This represents a standard
RL approach to game-playing without CGT-informed structure.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from collections import deque
from typing import Tuple, List, Optional


class DQNAgent:
    """DQN agent for Nim with experience replay and target network."""

    def __init__(
        self,
        max_heaps: int = 6,
        max_heap_size: int = 15,
        hidden_units: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
    ):
        self.max_heaps = max_heaps
        self.max_heap_size = max_heap_size
        self.n_actions = max_heaps * max_heap_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.rng = np.random.default_rng(42)

        self.q_network = self._build_network(hidden_units)
        self.target_network = self._build_network(hidden_units)
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self._update_target()

        self.replay_buffer = deque(maxlen=buffer_size)
        self.train_step_count = 0

    def _build_network(self, hidden_units: int) -> keras.Model:
        inputs = keras.Input(shape=(self.max_heaps,))
        x = layers.Dense(hidden_units, activation="relu")(inputs)
        x = layers.Dense(hidden_units, activation="relu")(x)
        q_values = layers.Dense(self.n_actions)(x)
        return keras.Model(inputs=inputs, outputs=q_values)

    def _update_target(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def _state_to_input(self, heaps: Tuple[int, ...]) -> np.ndarray:
        x = np.zeros(self.max_heaps, dtype=np.float32)
        x[: len(heaps)] = heaps
        return x

    def _action_to_move(self, action: int) -> Tuple[int, int]:
        heap_idx = action // self.max_heap_size
        remove = (action % self.max_heap_size) + 1
        return (heap_idx, remove)

    def _move_to_action(self, move: Tuple[int, int]) -> int:
        return move[0] * self.max_heap_size + (move[1] - 1)

    def _get_legal_action_mask(self, heaps: Tuple[int, ...]) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=np.float32)
        for i, h in enumerate(heaps):
            for r in range(1, h + 1):
                mask[i * self.max_heap_size + (r - 1)] = 1.0
        return mask

    def select_action(
        self, heaps: Tuple[int, ...], legal_moves: List[Tuple[int, int]]
    ) -> Tuple[int, int]:
        if not legal_moves:
            return (0, 1)

        if self.rng.random() < self.epsilon:
            return legal_moves[self.rng.integers(len(legal_moves))]

        state = self._state_to_input(heaps)[np.newaxis, :]
        q_vals = self.q_network(state, training=False).numpy()[0]

        legal_mask = self._get_legal_action_mask(heaps)
        q_vals = np.where(legal_mask > 0, q_vals, -1e9)
        best_action = int(np.argmax(q_vals))
        return self._action_to_move(best_action)

    def store_transition(
        self,
        state: Tuple[int, ...],
        action: Tuple[int, int],
        reward: float,
        next_state: Tuple[int, ...],
        done: bool,
    ):
        self.replay_buffer.append(
            (
                self._state_to_input(state),
                self._move_to_action(action),
                reward,
                self._state_to_input(next_state),
                done,
                self._get_legal_action_mask(next_state),
            )
        )

    def train_batch(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        indices = self.rng.integers(len(self.replay_buffer), size=self.batch_size)
        batch = [self.replay_buffer[i] for i in indices]

        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch], dtype=np.float32)
        next_legal = np.array([b[5] for b in batch])

        next_q = self.target_network(next_states, training=False).numpy()
        next_q = np.where(next_legal > 0, next_q, -1e9)
        next_q_max = np.max(next_q, axis=1)
        targets = rewards + (1 - dones) * self.gamma * next_q_max

        with tf.GradientTape() as tape:
            q_values = self.q_network(states, training=True)
            action_masks = tf.one_hot(actions, self.n_actions)
            q_selected = tf.reduce_sum(q_values * action_masks, axis=1)
            loss = tf.reduce_mean(tf.square(targets - q_selected))

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.q_network.trainable_variables)
        )

        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self._update_target()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(loss)

    def predict_win_loss(self, heaps: Tuple[int, ...]) -> float:
        """Estimate win probability: proportion of legal actions with positive Q."""
        state = self._state_to_input(heaps)[np.newaxis, :]
        q_vals = self.q_network(state, training=False).numpy()[0]
        legal_mask = self._get_legal_action_mask(heaps)
        legal_qs = q_vals[legal_mask > 0]
        if len(legal_qs) == 0:
            return 0.0
        return float(np.max(legal_qs) > 0)
