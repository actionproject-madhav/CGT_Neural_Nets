"""
Evaluation pipeline for the Nim pilot study.

Computes all metrics across all test sets for each architecture:
  - Win/Loss classification accuracy (ID and OOD)
  - Grundy value prediction accuracy (CGT-Net only)
  - Optimal move percentage
  - Samples-to-95%-accuracy (from training history)
"""

from __future__ import annotations
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Tuple, Optional

from games.nim import compute_grundy_value, get_optimal_moves, is_winning_position


class PilotEvaluator:
    """Evaluates all trained models across all test sets."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir

    def evaluate_supervised_model(
        self,
        model: keras.Model,
        model_name: str,
        test_datasets: Dict[str, Dict[str, np.ndarray]],
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate a supervised model on all test sets."""
        results = {}
        is_cgt = model_name == "cgt_net"

        for ds_name, data in test_datasets.items():
            if model_name == "mlp_baseline":
                inputs = data["positions"]
            else:
                inputs = [data["positions"], data["masks"]]

            if is_cgt:
                wl_pred, grundy_pred = model(inputs, training=False)
            else:
                wl_pred = model(inputs, training=False)

            wl_preds = tf.squeeze(wl_pred, axis=-1).numpy()
            wl_binary = (wl_preds > 0.5).astype(float)
            wl_acc = float(np.mean(wl_binary == data["win_loss"]))

            ds_results = {"win_loss_accuracy": wl_acc}

            if is_cgt:
                grundy_preds = np.argmax(grundy_pred.numpy(), axis=-1)
                mask = data["masks"]
                correct = (grundy_preds == data["per_heap_grundy"]).astype(float)
                masked_correct = correct * mask
                grundy_acc = float(
                    np.sum(masked_correct) / (np.sum(mask) + 1e-8)
                )
                ds_results["grundy_accuracy"] = grundy_acc

            results[ds_name] = ds_results

        return results

    def evaluate_dqn_agent(
        self,
        agent,
        test_datasets: Dict[str, Dict[str, np.ndarray]],
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate DQN agent on all test sets using Q-value based win/loss prediction."""
        results = {}

        for ds_name, data in test_datasets.items():
            correct = 0
            total = len(data["win_loss"])

            for i in range(total):
                n_heaps = int(data["num_heaps"][i])
                heaps = tuple(int(x) for x in data["positions"][i, :n_heaps])
                pred = agent.predict_win_loss(heaps)
                actual = data["win_loss"][i]
                if (pred > 0.5) == (actual > 0.5):
                    correct += 1

            results[ds_name] = {"win_loss_accuracy": correct / total}

        return results

    def compute_optimal_move_rate(
        self,
        model: keras.Model,
        model_name: str,
        test_data: Dict[str, np.ndarray],
        max_heaps: int = 6,
    ) -> float:
        """
        For winning positions, check if the model's top predicted action
        leads to a Grundy-0 successor (an optimal move).
        Supervised models predict win/loss, so we evaluate move quality
        by checking if moving to the lowest-valued successor works.
        """
        is_cgt = model_name == "cgt_net"
        optimal_count = 0
        winning_count = 0

        for i in range(len(test_data["win_loss"])):
            if test_data["win_loss"][i] < 0.5:
                continue

            n_heaps = int(test_data["num_heaps"][i])
            heaps = tuple(int(x) for x in test_data["positions"][i, :n_heaps])
            opt_moves = get_optimal_moves(heaps)

            if not opt_moves:
                continue

            winning_count += 1

            best_move = None
            best_score = float("inf")
            for hi in range(n_heaps):
                for rem in range(1, int(heaps[hi]) + 1):
                    new_heaps = list(heaps)
                    new_heaps[hi] -= rem
                    new_heaps_t = tuple(new_heaps)

                    padded = np.zeros(max_heaps, dtype=np.float32)
                    padded[:n_heaps] = new_heaps_t
                    mask = np.zeros(max_heaps, dtype=np.float32)
                    mask[:n_heaps] = 1.0

                    if model_name == "mlp_baseline":
                        inp = padded[np.newaxis, :]
                    else:
                        inp = [padded[np.newaxis, :], mask[np.newaxis, :]]

                    if is_cgt:
                        pred, _ = model(inp, training=False)
                    else:
                        pred = model(inp, training=False)

                    score = float(tf.squeeze(pred).numpy())
                    if score < best_score:
                        best_score = score
                        best_move = (hi, rem)

            if best_move in opt_moves:
                optimal_count += 1

        return optimal_count / max(winning_count, 1)

    def samples_to_threshold(
        self, history: Dict[str, List[float]], threshold: float = 0.95
    ) -> Optional[int]:
        """Return the epoch at which validation accuracy first exceeds threshold."""
        for i, acc in enumerate(history.get("val_acc", [])):
            if acc >= threshold:
                return i + 1
        return None

    def aggregate_multi_seed_results(
        self, all_seed_results: List[Dict[str, Dict[str, float]]]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Aggregate results across seeds: compute mean and std."""
        if not all_seed_results:
            return {}

        aggregated = {}
        for ds_name in all_seed_results[0]:
            aggregated[ds_name] = {}
            for metric in all_seed_results[0][ds_name]:
                values = [r[ds_name][metric] for r in all_seed_results]
                aggregated[ds_name][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "values": values,
                }

        return aggregated

    def save_results(self, all_results: Dict, filename: str = "pilot_results.json"):
        """Save all results to JSON."""
        path = os.path.join(self.results_dir, filename)
        with open(path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {path}")

    def print_comparison_table(self, all_results: Dict):
        """Print a formatted comparison table of all architectures."""
        test_sets = ["test_id", "test_ood_heaps", "test_ood_sizes", "test_ood_both"]
        headers = ["Model", "ID Acc", "OOD-Heaps", "OOD-Sizes", "OOD-Both"]

        print(f"\n{'='*75}")
        print("PILOT STUDY RESULTS: Win/Loss Classification Accuracy")
        print(f"{'='*75}")
        print(f"{'Model':<20} {'ID':>10} {'OOD-Heaps':>12} {'OOD-Sizes':>12} {'OOD-Both':>12}")
        print("-" * 75)

        for model_name, model_results in all_results.items():
            row = f"{model_name:<20}"
            for ds in test_sets:
                if ds in model_results:
                    stats = model_results[ds].get("win_loss_accuracy", {})
                    if isinstance(stats, dict):
                        mean = stats.get("mean", 0)
                        std = stats.get("std", 0)
                        row += f" {mean:.1%}±{std:.1%}"
                    else:
                        row += f" {stats:>10.1%}  "
                else:
                    row += f" {'N/A':>10}  "
            print(row)

        print(f"{'='*75}")
