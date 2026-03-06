"""
MLP Baseline: standard feedforward network for Nim win/loss prediction.

Input is a fixed-size vector of integer heap sizes (zero-padded to max_heaps).
Each heap value is embedded, then all embeddings are flattened into a single
vector for the feedforward layers.

This architecture treats heap positions as ordered and cannot generalize
to a different number of heaps without retraining.
"""

import keras
from keras import layers


def build_mlp_model(
    max_heaps: int = 6,
    vocab_size: int = 16,
    embed_dim: int = 16,
    hidden_units: int = 256,
    n_hidden: int = 4,
    dropout_rate: float = 0.1,
) -> keras.Model:
    inputs = keras.Input(shape=(max_heaps,), dtype="int32", name="heap_sizes")

    x = layers.Embedding(vocab_size, embed_dim, name="heap_embed")(inputs)
    x = layers.Flatten(name="flatten")(x)

    for i in range(n_hidden):
        x = layers.Dense(hidden_units, activation="relu", name=f"hidden_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i}")(x)

    win_loss = layers.Dense(1, activation="sigmoid", name="win_loss")(x)

    model = keras.Model(inputs=inputs, outputs=win_loss, name="mlp_baseline")
    return model
