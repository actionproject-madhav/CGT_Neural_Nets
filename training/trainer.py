"""
Supervised training loop for Nim pilot study.

Supports training MLP, DeepSets, and CGT-Net architectures with:
  - Early stopping with patience
  - Multi-seed runs for statistical significance
  - Checkpoint saving of best model
  - Training history logging
"""

from __future__ import annotations
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Tuple, Optional, Callable
from tqdm import tqdm


class SupervisedTrainer:
    """Unified trainer for all supervised architectures."""

    def __init__(
        self,
        model_builder: Callable,
        model_name: str,
        results_dir: str = "results",
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 100,
        patience: int = 10,
    ):
        self.model_builder = model_builder
        self.model_name = model_name
        self.results_dir = results_dir
        self.lr = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience

        os.makedirs(os.path.join(results_dir, model_name), exist_ok=True)

    def _prepare_inputs(
        self, data: Dict[str, np.ndarray], model_name: str
    ) -> Tuple:
        """Prepare model-specific inputs from the dataset dict."""
        if model_name == "mlp_baseline":
            return data["positions"]
        else:
            return [data["positions"], data["masks"]]

    def train_single_seed(
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Dict[str, np.ndarray],
        seed: int,
    ) -> Dict:
        """Train a single model with a given random seed."""
        tf.random.set_seed(seed)
        np.random.seed(seed)

        model = self.model_builder()
        is_cgt = self.model_name == "cgt_net"

        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.lr,
            decay_steps=self.max_epochs * (len(train_data["win_loss"]) // self.batch_size + 1),
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

        train_inputs = self._prepare_inputs(train_data, self.model_name)
        val_inputs = self._prepare_inputs(val_data, self.model_name)

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_weights = None

        for epoch in range(self.max_epochs):
            # --- Training ---
            epoch_losses = []
            epoch_correct = 0
            epoch_total = 0

            indices = np.random.permutation(len(train_data["win_loss"]))
            for start in range(0, len(indices), self.batch_size):
                batch_idx = indices[start : start + self.batch_size]

                if isinstance(train_inputs, list):
                    batch_x = [x[batch_idx] for x in train_inputs]
                else:
                    batch_x = train_inputs[batch_idx]

                batch_y_wl = train_data["win_loss"][batch_idx]
                batch_y_grundy = train_data["per_heap_grundy"][batch_idx]

                with tf.GradientTape() as tape:
                    if is_cgt:
                        wl_pred, grundy_pred = model(batch_x, training=True)
                        wl_loss = keras.losses.binary_crossentropy(
                            batch_y_wl, tf.squeeze(wl_pred, axis=-1)
                        )

                        grundy_loss = keras.losses.sparse_categorical_crossentropy(
                            batch_y_grundy, grundy_pred
                        )
                        mask = train_data["masks"][batch_idx]
                        grundy_loss = tf.reduce_sum(grundy_loss * mask, axis=-1) / (
                            tf.reduce_sum(mask, axis=-1) + 1e-8
                        )

                        loss = tf.reduce_mean(wl_loss) + 0.3 * tf.reduce_mean(grundy_loss)
                    else:
                        wl_pred = model(batch_x, training=True)
                        loss = tf.reduce_mean(
                            keras.losses.binary_crossentropy(
                                batch_y_wl, tf.squeeze(wl_pred, axis=-1)
                            )
                        )

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                epoch_losses.append(float(loss))
                preds = (tf.squeeze(wl_pred if not is_cgt else wl_pred, axis=-1).numpy() > 0.5).astype(float)
                epoch_correct += np.sum(preds == batch_y_wl)
                epoch_total += len(batch_y_wl)

            train_loss = np.mean(epoch_losses)
            train_acc = epoch_correct / epoch_total

            # --- Validation ---
            if is_cgt:
                val_wl_pred, val_grundy_pred = model(val_inputs, training=False)
            else:
                val_wl_pred = model(val_inputs, training=False)

            val_preds = tf.squeeze(val_wl_pred, axis=-1).numpy()
            val_loss_arr = keras.losses.binary_crossentropy(
                val_data["win_loss"], val_preds
            )
            val_loss = float(tf.reduce_mean(val_loss_arr))
            val_acc = float(np.mean((val_preds > 0.5).astype(float) == val_data["win_loss"]))

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            # --- Early stopping ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = model.get_weights()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_weights is not None:
            model.set_weights(best_weights)

        save_dir = os.path.join(self.results_dir, self.model_name, f"seed_{seed}")
        os.makedirs(save_dir, exist_ok=True)
        model.save(os.path.join(save_dir, "model.keras"))
        with open(os.path.join(save_dir, "history.json"), "w") as f:
            json.dump(history, f)

        return {
            "model": model,
            "history": history,
            "best_val_loss": best_val_loss,
            "final_epoch": len(history["train_loss"]),
            "seed": seed,
        }

    def train_multi_seed(
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Dict[str, np.ndarray],
        seeds: List[int],
    ) -> List[Dict]:
        """Train across multiple seeds and return all results."""
        results = []
        for seed in seeds:
            print(f"\n{'='*50}")
            print(f"Training {self.model_name} | Seed {seed}")
            print(f"{'='*50}")
            result = self.train_single_seed(train_data, val_data, seed)
            print(
                f"  Final epoch: {result['final_epoch']} | "
                f"Best val loss: {result['best_val_loss']:.4f} | "
                f"Val acc: {result['history']['val_acc'][-1]:.4f}"
            )
            results.append(result)
        return results
