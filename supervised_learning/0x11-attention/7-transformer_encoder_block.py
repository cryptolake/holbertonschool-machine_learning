#!/usr/bin/env python3
"""Transformer encoder block."""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """Transformer encoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize instance."""
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Functionality of the layer."""
        mha_out, _ = self.mha(x, x, x, mask)
        norm1 = self.layernorm1(mha_out + x, training=training)
        y1 = self.dense_hidden(self.dropout1(norm1, training=training),
                               training=training)
        y2 = self.dense_output(self.dropout2(y1, training=training),
                               training=training)
        norm2 = self.layernorm2(norm1 + y2, training=training)
        return norm2
