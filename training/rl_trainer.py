"""
DQN self-play training loop for Nim.

Trains a DQN agent by playing games against itself (or a random opponent).
The agent alternates between acting as player 0 and player 1.
"""

from __future__ import annotations
import os
import json
import numpy as np
from typing import Dict, List, Tuple
from tqdm import trange

from games.nim import NimGame, get_optimal_moves
from models.dqn_baseline import DQNAgent


class DQNTrainer:
    """Trains a DQN agent on Nim via self-play."""

    def __init__(
        self,
        max_heaps: int = 6,
        max_heap_size: int = 15,
        results_dir: str = "results",
    ):
        self.max_heaps = max_heaps
        self.max_heap_size = max_heap_size
        self.results_dir = results_dir
        os.makedirs(os.path.join(results_dir, "dqn"), exist_ok=True)

    def _random_starting_position(
        self,
        rng: np.random.Generator,
        heap_counts: List[int],
        heap_size_range: Tuple[int, int],
    ) -> Tuple[int, ...]:
        n_heaps = rng.choice(heap_counts)
        lo, hi = heap_size_range
        heaps = tuple(int(x) for x in rng.integers(lo, hi + 1, size=n_heaps))
        return heaps

    def train_single_seed(
        self,
        seed: int,
        n_episodes: int = 50000,
        heap_counts: List[int] = None,
        heap_size_range: Tuple[int, int] = (0, 7),
        log_interval: int = 1000,
    ) -> Dict:
        if heap_counts is None:
            heap_counts = [2, 3]

        rng = np.random.default_rng(seed)
        agent = DQNAgent(
            max_heaps=self.max_heaps,
            max_heap_size=self.max_heap_size,
        )
        agent.rng = rng

        history = {"episode_rewards": [], "losses": [], "epsilons": [], "win_rates": []}
        recent_wins = []

        for ep in trange(n_episodes, desc=f"DQN seed={seed}", leave=False):
            heaps = self._random_starting_position(rng, heap_counts, heap_size_range)

            if all(h == 0 for h in heaps):
                continue

            game = NimGame(heaps)
            total_reward = 0
            transitions = []

            while not game.done:
                state = game.get_state()
                legal = game.get_legal_moves()

                if not legal:
                    break

                if game.current_player == 0:
                    action = agent.select_action(state, legal)
                else:
                    action = legal[rng.integers(len(legal))]

                next_state, reward, done = game.step(action)

                if game.current_player == 0 or done:
                    agent_reward = reward if game.winner == 0 else -reward if done else 0
                    transitions.append((state, action, agent_reward, next_state, done))
                    total_reward += agent_reward

            for t in transitions:
                agent.store_transition(*t)
            loss = agent.train_batch()

            recent_wins.append(1 if game.winner == 0 else 0)
            if len(recent_wins) > 1000:
                recent_wins.pop(0)

            if loss is not None:
                history["losses"].append(float(loss))
            history["epsilons"].append(agent.epsilon)

            if (ep + 1) % log_interval == 0:
                wr = np.mean(recent_wins) if recent_wins else 0
                history["win_rates"].append(float(wr))

        save_dir = os.path.join(self.results_dir, "dqn", f"seed_{seed}")
        os.makedirs(save_dir, exist_ok=True)
        agent.q_network.save(os.path.join(save_dir, "q_network.keras"))
        with open(os.path.join(save_dir, "history.json"), "w") as f:
            json.dump(history, f)

        return {"agent": agent, "history": history, "seed": seed}

    def train_multi_seed(
        self,
        seeds: List[int],
        n_episodes: int = 50000,
        heap_counts: List[int] = None,
        heap_size_range: Tuple[int, int] = (0, 7),
    ) -> List[Dict]:
        results = []
        for seed in seeds:
            print(f"\n{'='*50}")
            print(f"Training DQN | Seed {seed}")
            print(f"{'='*50}")
            result = self.train_single_seed(
                seed, n_episodes, heap_counts, heap_size_range
            )
            wr = result["history"]["win_rates"][-1] if result["history"]["win_rates"] else 0
            print(f"  Final win rate vs random: {wr:.4f}")
            results.append(result)
        return results
