"""
Data generator for Nim pilot study.

Generates labeled Nim positions for supervised training and evaluation.
Produces 6 dataset splits:
  - Train / Val / Test-ID: in-distribution (2-3 heaps, sizes 0-7)
  - Test-OOD-Heaps: out-of-distribution (4-6 heaps, sizes 0-7)
  - Test-OOD-Sizes: out-of-distribution (2-3 heaps, sizes 8-15)
  - Test-OOD-Both: out-of-distribution (4-6 heaps, sizes 8-15)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, List
from itertools import product

from games.nim import compute_grundy_value, is_winning_position, get_optimal_moves


class NimDataGenerator:
    """Generates and manages all datasets for the pilot study."""

    def __init__(self, max_pad_heaps: int = 6, seed: int = 42):
        self.max_pad_heaps = max_pad_heaps
        self.rng = np.random.default_rng(seed)

    def _generate_positions(
        self,
        heap_counts: List[int],
        heap_size_range: Tuple[int, int],
    ) -> List[Tuple[int, ...]]:
        """Generate all Nim positions for given heap counts and size range."""
        lo, hi = heap_size_range
        all_positions = []
        for n_heaps in heap_counts:
            ranges = [range(lo, hi + 1)] * n_heaps
            all_positions.extend(product(*ranges))
        return all_positions

    def _label_positions(
        self, positions: List[Tuple[int, ...]]
    ) -> Dict[str, np.ndarray]:
        """
        Label positions with Grundy values, win/loss, and optimal move existence.
        Returns arrays padded to max_pad_heaps.
        """
        n = len(positions)
        X = np.zeros((n, self.max_pad_heaps), dtype=np.float32)
        masks = np.zeros((n, self.max_pad_heaps), dtype=np.float32)
        grundy = np.zeros(n, dtype=np.int32)
        win_loss = np.zeros(n, dtype=np.float32)
        n_heaps = np.zeros(n, dtype=np.int32)
        per_heap_grundy = np.zeros((n, self.max_pad_heaps), dtype=np.int32)

        for i, heaps in enumerate(positions):
            k = len(heaps)
            X[i, :k] = heaps
            masks[i, :k] = 1.0
            grundy[i] = compute_grundy_value(heaps)
            win_loss[i] = 1.0 if is_winning_position(heaps) else 0.0
            n_heaps[i] = k
            for j, h in enumerate(heaps):
                per_heap_grundy[i, j] = h  # Grundy value of a Nim heap = heap size

        return {
            "positions": X,
            "masks": masks,
            "grundy_values": grundy,
            "win_loss": win_loss,
            "num_heaps": n_heaps,
            "per_heap_grundy": per_heap_grundy,
            "raw_positions": positions,
        }

    def generate_all_datasets(
        self,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Generate all 6 datasets for the pilot study."""
        datasets = {}

        # In-distribution: 2-3 heaps, sizes 0-7
        id_positions = self._generate_positions([2, 3], (0, 7))
        self.rng.shuffle(id_positions)

        n = len(id_positions)
        n_train = int(n * train_val_test_split[0])
        n_val = int(n * train_val_test_split[1])

        datasets["train"] = self._label_positions(id_positions[:n_train])
        datasets["val"] = self._label_positions(id_positions[n_train : n_train + n_val])
        datasets["test_id"] = self._label_positions(id_positions[n_train + n_val :])

        # OOD-Heaps: 4-6 heaps, sizes 0-7
        ood_heaps_positions = self._generate_positions([4, 5, 6], (0, 7))
        self.rng.shuffle(ood_heaps_positions)
        ood_heaps_sample = ood_heaps_positions[: min(10000, len(ood_heaps_positions))]
        datasets["test_ood_heaps"] = self._label_positions(ood_heaps_sample)

        # OOD-Sizes: 2-3 heaps, sizes 8-15
        ood_sizes_positions = self._generate_positions([2, 3], (8, 15))
        self.rng.shuffle(ood_sizes_positions)
        datasets["test_ood_sizes"] = self._label_positions(ood_sizes_positions)

        # OOD-Both: 4-6 heaps, sizes 8-15
        ood_both_positions = self._generate_positions([4, 5, 6], (8, 15))
        self.rng.shuffle(ood_both_positions)
        ood_both_sample = ood_both_positions[: min(10000, len(ood_both_positions))]
        datasets["test_ood_both"] = self._label_positions(ood_both_sample)

        return datasets

    def get_dataset_stats(self, datasets: Dict[str, Dict[str, np.ndarray]]) -> str:
        """Print summary statistics for all datasets."""
        lines = ["Dataset Statistics", "=" * 60]
        for name, data in datasets.items():
            n = len(data["win_loss"])
            win_frac = data["win_loss"].mean()
            lines.append(
                f"  {name:20s}: {n:7d} positions | "
                f"Win: {win_frac:.1%} | Loss: {1 - win_frac:.1%}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)
