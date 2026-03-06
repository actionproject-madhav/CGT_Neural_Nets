"""
Data generator for Nim pilot study.

Generates labeled Nim positions for supervised training and evaluation.
Produces 6 dataset splits:
  - Train / Val / Test-ID: in-distribution (2-4 heaps, sizes 0-7)
  - Test-OOD-Heaps: out-of-distribution (5-6 heaps, sizes 0-7)
  - Test-OOD-Sizes: out-of-distribution (2-4 heaps, sizes 8-15)
  - Test-OOD-Both: out-of-distribution (5-6 heaps, sizes 8-15)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, List
from itertools import product

from games.nim import compute_grundy_value, is_winning_position, get_optimal_moves


class NimDataGenerator:
    """Generates and manages all datasets for the pilot study."""

    def __init__(self, max_pad_heaps: int = 6, max_heap_value: int = 15, seed: int = 42):
        self.max_pad_heaps = max_pad_heaps
        self.max_heap_value = max_heap_value
        self.rng = np.random.default_rng(seed)

    def _generate_positions(
        self,
        heap_counts: List[int],
        heap_size_range: Tuple[int, int],
        max_positions: int = None,
    ) -> List[Tuple[int, ...]]:
        """Generate Nim positions. Enumerates all if feasible, else samples."""
        lo, hi = heap_size_range
        all_positions = []
        for n_heaps in heap_counts:
            total = (hi - lo + 1) ** n_heaps
            if max_positions is not None and total > max_positions * 2:
                for _ in range(max_positions // len(heap_counts)):
                    pos = tuple(int(x) for x in self.rng.integers(lo, hi + 1, size=n_heaps))
                    all_positions.append(pos)
            else:
                ranges = [range(lo, hi + 1)] * n_heaps
                all_positions.extend(product(*ranges))

        self.rng.shuffle(all_positions)
        if max_positions is not None and len(all_positions) > max_positions:
            all_positions = all_positions[:max_positions]
        return all_positions

    def _label_positions(
        self, positions: List[Tuple[int, ...]]
    ) -> Dict[str, np.ndarray]:
        """
        Label positions with Grundy values, win/loss, and per-heap Grundy values.
        Heap sizes stored as integers for use with embedding layers.
        """
        n = len(positions)
        X = np.zeros((n, self.max_pad_heaps), dtype=np.int32)
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
                per_heap_grundy[i, j] = min(h, self.max_heap_value)

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
        train_heap_counts: List[int] = None,
        train_heap_range: Tuple[int, int] = (0, 7),
        ood_heap_counts: List[int] = None,
        ood_size_range: Tuple[int, int] = (8, 15),
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        ood_sample_limit: int = 10000,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Generate all 6 datasets for the pilot study."""
        if train_heap_counts is None:
            train_heap_counts = [2, 3, 4]
        if ood_heap_counts is None:
            ood_heap_counts = [5, 6]

        datasets = {}

        id_positions = self._generate_positions(train_heap_counts, train_heap_range)
        self.rng.shuffle(id_positions)

        n = len(id_positions)
        n_train = int(n * train_val_test_split[0])
        n_val = int(n * train_val_test_split[1])

        datasets["train"] = self._label_positions(id_positions[:n_train])
        datasets["val"] = self._label_positions(id_positions[n_train : n_train + n_val])
        datasets["test_id"] = self._label_positions(id_positions[n_train + n_val :])

        ood_heaps_positions = self._generate_positions(
            ood_heap_counts, train_heap_range, max_positions=ood_sample_limit
        )
        datasets["test_ood_heaps"] = self._label_positions(ood_heaps_positions)

        ood_sizes_positions = self._generate_positions(
            train_heap_counts, ood_size_range, max_positions=ood_sample_limit
        )
        datasets["test_ood_sizes"] = self._label_positions(ood_sizes_positions)

        ood_both_positions = self._generate_positions(
            ood_heap_counts, ood_size_range, max_positions=ood_sample_limit
        )
        datasets["test_ood_both"] = self._label_positions(ood_both_positions)

        return datasets

    def balance_dataset(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Oversample the minority class to create a balanced training set."""
        labels = data["win_loss"]
        pos_idx = np.where(labels > 0.5)[0]
        neg_idx = np.where(labels <= 0.5)[0]

        if len(pos_idx) > len(neg_idx):
            oversample_idx = self.rng.choice(neg_idx, size=len(pos_idx), replace=True)
            balanced_idx = np.concatenate([pos_idx, oversample_idx])
        else:
            oversample_idx = self.rng.choice(pos_idx, size=len(neg_idx), replace=True)
            balanced_idx = np.concatenate([neg_idx, oversample_idx])

        self.rng.shuffle(balanced_idx)

        balanced = {}
        for key, val in data.items():
            if key == "raw_positions":
                balanced[key] = [val[i] for i in balanced_idx]
            elif isinstance(val, np.ndarray):
                balanced[key] = val[balanced_idx]
            else:
                balanced[key] = val
        return balanced

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
