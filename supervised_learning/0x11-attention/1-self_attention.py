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
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Layer functionality."""
        s_prev = tf.expand_dims(s_prev, 1)
        o_W = self.W(s_prev)
        o_U = self.U(hidden_states)
        energy = self.V(
            tf.nn.tanh(o_W + o_U)
        )
        attention = tf.nn.softmax(energy, axis=1)
        attention = tf.cast(attention, dtype=tf.float64)
        hidden_states = tf.cast(hidden_states, dtype=tf.float64)
        context = tf.reduce_sum(attention * hidden_states, axis=1)
        return context, attention
