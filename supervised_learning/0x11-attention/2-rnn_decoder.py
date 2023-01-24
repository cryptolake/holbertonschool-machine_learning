#!/usr/bin/env python3
"""RNN Decoder for machine translation."""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Decoder layer for Machine translation."""

    def __init__(self, vocab, embedding, units, batch):
        """Initialize Class."""
        super().__init__()

        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units, return_state=True,
                                       return_sequences=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)
        self.batch = batch
        self.units = units

    def call(self, x, s_prev, hidden_states):
        """Contains Layer functionality."""
        attention = SelfAttention(self.units)
        context, _ = attention(s_prev, hidden_states)
        x = self.embedding(x)
        context = tf.expand_dims(context, 1)
        X = tf.concat([context, x], axis=-1)
        y, hidden = self.gru(inputs=X)
        Y = tf.reshape(y, (-1, y.shape[2]))
        Y = self.F(Y)
        return Y, hidden
