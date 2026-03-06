"""
DeepSets Baseline: permutation-invariant network for Nim.

Architecture follows Zaheer et al. (2017) "Deep Sets":
  - Embedding: maps each discrete heap size to a learned vector
  - phi: per-element network applied independently to each heap embedding
  - sum aggregation (permutation invariant), masked for variable heap counts
  - rho: post-aggregation network producing final prediction

Handles variable numbers of heaps via masking. Permutation invariant
by construction, but does not encode CGT-specific algebraic structure.
"""

import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="deepsets")
class MaskedSumPool(layers.Layer):
    """Sum pooling over the heap dimension, respecting the mask."""

    def call(self, inputs):
        x, mask = inputs
        mask_expanded = ops.expand_dims(mask, axis=-1)
        return ops.sum(x * mask_expanded, axis=1)


def build_deepsets_model(
    max_heaps: int = 6,
    vocab_size: int = 16,
    embed_dim: int = 16,
    phi_units: int = 128,
    phi_layers: int = 3,
    rho_units: int = 128,
    rho_layers: int = 3,
    dropout_rate: float = 0.1,
) -> keras.Model:
    heap_input = keras.Input(shape=(max_heaps,), dtype="int32", name="heap_sizes")
    mask_input = keras.Input(shape=(max_heaps,), name="mask")

    x = layers.Embedding(vocab_size, embed_dim, name="heap_embed")(heap_input)

    for i in range(phi_layers):
        x = layers.TimeDistributed(
            layers.Dense(phi_units, activation="relu"), name=f"phi_{i}"
        )(x)

    x = MaskedSumPool(name="sum_pool")([x, mask_input])

    for i in range(rho_layers):
        x = layers.Dense(rho_units, activation="relu", name=f"rho_{i}")(x)
        x = layers.BatchNormalization(name=f"rho_bn_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"rho_drop_{i}")(x)

    win_loss = layers.Dense(1, activation="sigmoid", name="win_loss")(x)

    model = keras.Model(
        inputs=[heap_input, mask_input],
        outputs=win_loss,
        name="deepsets_baseline",
    )
    return model
