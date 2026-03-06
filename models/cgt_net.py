"""
CGT-Decomposition Network: novel architecture encoding combinatorial game theory structure.

Key design principles from CGT:
  1. Each heap is an independent subgame -- process with shared weights
  2. Subgames have Grundy values -- auxiliary loss to learn them
  3. Combined Grundy value = XOR of individual values -- learned aggregation
     should converge to an XOR-like operation

Architecture:
  - Subgame Encoder: shared network mapping each heap to a Grundy embedding
  - Auxiliary Grundy Head: per-heap Grundy value prediction (supervision signal)
  - CGT Aggregator: learned aggregation of embeddings (bitwise structure)
  - Outcome Head: final win/loss from aggregated embedding
"""

import keras
from keras import layers, ops


class BitwiseAggregator(layers.Layer):
    """
    Aggregation layer designed to be able to represent XOR-like operations.

    Uses a bank of binary-like features: each subgame encoder output is
    passed through sigmoid to get soft-binary representations, then
    aggregated in a way that allows learning parity (XOR).

    The key insight: XOR can be decomposed as (a + b) mod 2 for single bits.
    We use soft modular arithmetic via sin/cos features of the sum, which
    naturally capture periodicity.
    """

    def __init__(self, embed_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.project = layers.Dense(embed_dim * 2, activation="tanh")

    def call(self, inputs):
        x, mask = inputs
        mask_expanded = ops.expand_dims(mask, axis=-1)

        projected = self.project(x)
        masked = projected * mask_expanded
        summed = ops.sum(masked, axis=1)

        sin_features = ops.sin(summed[:, : self.embed_dim])
        cos_features = ops.cos(summed[:, self.embed_dim :])
        return ops.concatenate([sin_features, cos_features, summed], axis=-1)


def build_cgt_model(
    max_heaps: int = 6,
    embed_dim: int = 16,
    encoder_units: int = 64,
    encoder_layers: int = 2,
    max_grundy: int = 16,
    dropout_rate: float = 0.2,
) -> keras.Model:
    heap_input = keras.Input(shape=(max_heaps,), name="heap_sizes")
    mask_input = keras.Input(shape=(max_heaps,), name="mask")

    # --- Subgame Encoder (shared across heaps) ---
    x = layers.Reshape((max_heaps, 1))(heap_input)

    for i in range(encoder_layers):
        x = layers.TimeDistributed(
            layers.Dense(encoder_units, activation="relu"),
            name=f"encoder_{i}",
        )(x)

    grundy_embedding = layers.TimeDistributed(
        layers.Dense(embed_dim, activation="tanh"),
        name="grundy_embedding",
    )(x)

    # --- Auxiliary: per-heap Grundy value prediction ---
    grundy_logits = layers.TimeDistributed(
        layers.Dense(max_grundy, activation="softmax"),
        name="grundy_head",
    )(grundy_embedding)

    # --- CGT Aggregator ---
    aggregated = BitwiseAggregator(embed_dim=embed_dim, name="cgt_aggregator")(
        [grundy_embedding, mask_input]
    )

    # --- Outcome Head ---
    x = aggregated
    x = layers.Dense(64, activation="relu", name="outcome_hidden_0")(x)
    x = layers.Dropout(dropout_rate, name="outcome_drop_0")(x)
    x = layers.Dense(32, activation="relu", name="outcome_hidden_1")(x)

    win_loss = layers.Dense(1, activation="sigmoid", name="win_loss")(x)

    model = keras.Model(
        inputs=[heap_input, mask_input],
        outputs=[win_loss, grundy_logits],
        name="cgt_decomposition_net",
    )
    return model
