"""
Multi-heap Nim game engine with CGT-theoretic computations.

Nim rules: Two players alternate turns. On each turn, a player removes
one or more objects from a single heap. The player who takes the last
object wins (normal play convention).

By the Sprague-Grundy theorem, the Grundy value of a Nim position is
the XOR (nim-sum) of all heap sizes. A position is a losing position
(P-position) for the player to move iff the Grundy value is 0.
"""

from __future__ import annotations
import numpy as np
from itertools import product
from typing import List, Tuple, Optional
import functools


def compute_grundy_value(heaps: Tuple[int, ...]) -> int:
    """Compute the Grundy value of a Nim position (XOR of all heap sizes)."""
    return functools.reduce(lambda a, b: a ^ b, heaps, 0)


def is_winning_position(heaps: Tuple[int, ...]) -> bool:
    """A position is winning for the player to move iff Grundy value != 0."""
    return compute_grundy_value(heaps) != 0


def get_legal_moves(heaps: Tuple[int, ...]) -> List[Tuple[int, int]]:
    """
    Return all legal moves as (heap_index, objects_to_remove) pairs.
    A move removes 1..heap_size objects from a single heap.
    """
    moves = []
    for i, size in enumerate(heaps):
        for remove in range(1, size + 1):
            moves.append((i, remove))
    return moves


def apply_move(heaps: Tuple[int, ...], move: Tuple[int, int]) -> Tuple[int, ...]:
    """Apply a move and return the new heap configuration."""
    heap_idx, remove = move
    new_heaps = list(heaps)
    new_heaps[heap_idx] -= remove
    assert new_heaps[heap_idx] >= 0, "Cannot remove more than heap size"
    return tuple(new_heaps)


def get_optimal_moves(heaps: Tuple[int, ...]) -> List[Tuple[int, int]]:
    """
    Return all optimal moves from a winning position.
    An optimal move leads to a P-position (Grundy value 0).
    Returns empty list if the position is already losing.
    """
    if not is_winning_position(heaps):
        return []

    optimal = []
    for move in get_legal_moves(heaps):
        new_heaps = apply_move(heaps, move)
        if compute_grundy_value(new_heaps) == 0:
            optimal.append(move)
    return optimal


def is_terminal(heaps: Tuple[int, ...]) -> bool:
    """Game is over when all heaps are empty."""
    return all(h == 0 for h in heaps)


class NimGame:
    """Interactive Nim game environment for RL training."""

    def __init__(self, initial_heaps: Tuple[int, ...]):
        self.initial_heaps = initial_heaps
        self.heaps = list(initial_heaps)
        self.current_player = 0  # 0 or 1
        self.done = False
        self.winner = None

    def reset(self, heaps: Optional[Tuple[int, ...]] = None) -> Tuple[int, ...]:
        """Reset game to initial or given state."""
        if heaps is not None:
            self.initial_heaps = heaps
        self.heaps = list(self.initial_heaps)
        self.current_player = 0
        self.done = False
        self.winner = None
        return tuple(self.heaps)

    def step(self, move: Tuple[int, int]) -> Tuple[Tuple[int, ...], float, bool]:
        """
        Execute a move. Returns (new_state, reward, done).
        Reward: +1 for winning, -1 for losing, 0 otherwise.
        """
        heap_idx, remove = move
        assert not self.done, "Game is already over"
        assert 0 <= heap_idx < len(self.heaps), f"Invalid heap index {heap_idx}"
        assert 1 <= remove <= self.heaps[heap_idx], f"Invalid remove count {remove}"

        self.heaps[heap_idx] -= remove

        if is_terminal(tuple(self.heaps)):
            self.done = True
            self.winner = self.current_player
            reward = 1.0
        else:
            reward = 0.0
            self.current_player = 1 - self.current_player

        return tuple(self.heaps), reward, self.done

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        return get_legal_moves(tuple(self.heaps))

    def get_state(self) -> Tuple[int, ...]:
        return tuple(self.heaps)

    def clone(self) -> 'NimGame':
        game = NimGame(tuple(self.heaps))
        game.current_player = self.current_player
        game.done = self.done
        game.winner = self.winner
        return game


def generate_all_positions(num_heaps: int, max_heap_size: int) -> List[Tuple[int, ...]]:
    """Generate all possible Nim positions for given parameters."""
    ranges = [range(max_heap_size + 1)] * num_heaps
    return list(product(*ranges))


def play_optimal_vs_random(heaps: Tuple[int, ...], n_games: int = 100) -> float:
    """Play optimal strategy against random, return optimal win rate."""
    rng = np.random.default_rng(42)
    wins = 0
    for _ in range(n_games):
        game = NimGame(heaps)
        while not game.done:
            if game.current_player == 0:
                opt_moves = get_optimal_moves(game.get_state())
                if opt_moves:
                    move = opt_moves[rng.integers(len(opt_moves))]
                else:
                    moves = game.get_legal_moves()
                    move = moves[rng.integers(len(moves))]
            else:
                moves = game.get_legal_moves()
                move = moves[rng.integers(len(moves))]
            game.step(move)
        if game.winner == 0:
            wins += 1
    return wins / n_games
