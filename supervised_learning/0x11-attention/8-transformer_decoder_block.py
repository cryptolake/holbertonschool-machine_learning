#!/usr/bin/env python3
"""Transformer decoder block."""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """Transformer decoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize instance."""
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Functionality of the layer."""
        mha1, _ = self.mha1(x, x, x, look_ahead_mask)

        norm1 = self.layernorm1(self.dropout1(mha1, training=training) + x,
                                training=training)
        mha2, _ = self.mha2(norm1, encoder_output,
                            encoder_output, padding_mask)
        norm2 = self.layernorm2(self.dropout2(mha2, training=training) + norm1,
                                training=training)
        y1 = self.dense_hidden(norm2, training=training)
        y2 = self.dense_output(y1, training=training)

        norm3 = self.layernorm3(self.dropout3(y2, training=training) + norm2,
                                training=training)
        return norm3
