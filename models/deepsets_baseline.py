"""
DeepSets Baseline: permutation-invariant network for Nim.

Architecture follows Zaheer et al. (2017) "Deep Sets":
  - phi: per-element network applied independently to each heap
  - sum aggregation (permutation invariant)
  - rho: post-aggregation network producing final prediction

Handles variable numbers of heaps via masking. Permutation invariant
by construction, but does not encode CGT-specific algebraic structure.
"""

import keras
from keras import layers, ops


class MaskedSumPool(layers.Layer):
    """Sum pooling over the heap dimension, respecting the mask."""

    def call(self, inputs):
        x, mask = inputs
        mask_expanded = ops.expand_dims(mask, axis=-1)
        return ops.sum(x * mask_expanded, axis=1)


def build_deepsets_model(
    max_heaps: int = 6,
    phi_units: int = 64,
    phi_layers: int = 2,
    rho_units: int = 64,
    rho_layers: int = 2,
    dropout_rate: float = 0.2,
) -> keras.Model:
    heap_input = keras.Input(shape=(max_heaps,), name="heap_sizes")
    mask_input = keras.Input(shape=(max_heaps,), name="mask")

    x = layers.Reshape((max_heaps, 1))(heap_input)

    for i in range(phi_layers):
        x = layers.TimeDistributed(
            layers.Dense(phi_units, activation="relu"), name=f"phi_{i}"
        )(x)

    x = MaskedSumPool(name="sum_pool")([x, mask_input])

    for i in range(rho_layers):
        x = layers.Dense(rho_units, activation="relu", name=f"rho_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"rho_drop_{i}")(x)

    win_loss = layers.Dense(1, activation="sigmoid", name="win_loss")(x)

    model = keras.Model(
        inputs=[heap_input, mask_input],
        outputs=win_loss,
        name="deepsets_baseline",
    )
    return model
