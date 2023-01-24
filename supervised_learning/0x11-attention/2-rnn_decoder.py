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
        self.attention = SelfAttention(self.units)

    def call(self, x, s_prev, hidden_states):
        """Contains Layer functionality."""
        context, _ = self.attention(s_prev, hidden_states)
        x = self.embedding(x)
        X = tf.concat([tf.expand_dims(context, 1), x], axis=-1)

        output, state = self.gru(X)
        output = tf.reshape(output, (-1, output.shape[2]))

        Y = self.F(output)
        return Y, state
