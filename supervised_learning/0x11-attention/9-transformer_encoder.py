#!/usr/bin/env python3
"""Transformer encoder stack."""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """Transformer encoder stack."""

    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """Initialize class."""
        super().__init__()
        self.N = N
        self.dm = dm

        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """Call the layer."""
        input_len = x.shape[1]
        p_e = self.positional_encoding[:input_len]
        x = self.embedding(x)

        # This is the reason we scale it:
        # https://stackoverflow.com/questions/56930821/why-does-embedding-vector-multiplied-by-a-constant-in-transformer-model
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        x = x + p_e
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training, mask)
        return x
