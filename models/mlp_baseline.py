"""
MLP Baseline: standard feedforward network for Nim win/loss prediction.

Input is a fixed-size vector of heap sizes, zero-padded to max_heaps.
This architecture treats heap positions as ordered and cannot generalize
to a different number of heaps without retraining.
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers


def build_mlp_model(
    max_heaps: int = 6,
    hidden_units: int = 128,
    n_hidden: int = 3,
    dropout_rate: float = 0.2,
) -> keras.Model:
    inputs = keras.Input(shape=(max_heaps,), name="heap_sizes")

    x = inputs
    for i in range(n_hidden):
        x = layers.Dense(hidden_units, activation="relu", name=f"hidden_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i}")(x)

    win_loss = layers.Dense(1, activation="sigmoid", name="win_loss")(x)

    model = keras.Model(inputs=inputs, outputs=win_loss, name="mlp_baseline")
    return model
