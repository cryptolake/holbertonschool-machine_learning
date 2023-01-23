#!/usr/bin/env python3
"""Self Attention layer keras."""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Keras self Attention layer.

    Class constructor def __init__(self, units):
        units is an integer representing the number
        of hidden units in the alignment model
        Sets the following public instance attributes:
            W - a Dense layer with units units,
            to be applied to the previous decoder hidden state
            U - a Dense layer with units units,
            to be applied to the encoder hidden states
            V - a Dense layer with 1 units,
            to be applied to the tanh of the sum of the outputs of W and U
    Public instance method def call(self, s_prev, hidden_states):
        s_prev is a tensor of shape (batch, units)
        containing the previous decoder hidden state
        hidden_states is a tensor of shape
        (batch, input_seq_len, units)containing the outputs of the encoder
        Returns: context, weights
            context is a tensor of shape (batch, units) that contains
            the context vector for the decoder
            weights is a tensor of shape (batch, input_seq_len, 1)
            that contains the attention weights
    """

    def __init__(self, units):
        """Initialize SelfAttention."""
        super().__init__()
        self.W = tf.keras.layers.Dense(units, activation='tanh')
        self.U = tf.keras.layers.Dense(units, activation='tanh')
        self.V = tf.keras.layers.Dense(1, activation='softmax')

    def call(self, s_prev, hidden_states):
        """Layer functionality."""
        b, h_s, _ = hidden_states.shape
        o_W = self.W(s_prev)
        o_U = self.U(hidden_states)

        energy = []
        for i in range(h_s):
            conca = tf.concat([o_U[:, i, :], o_W], axis=1)
            energy.append(tf.reshape(conca, shape=(b, 1, conca.shape[-1])))
        energy = tf.concat(energy, axis=1)
        attention = tf.cast(self.V(energy), dtype=tf.float64)
        hidden_states = tf.cast(hidden_states, dtype=tf.float64)
        context = tf.reduce_sum(tf.multiply(attention, hidden_states), axis=1)

        return context, attention
