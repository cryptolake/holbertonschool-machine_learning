#!/usr/bin/env python3
"""Decoder for machine translation keras."""
import tensorflow as tf


class RNNDecoder(tf.keras.layers.Layer):
    """
    Decoder layer for Machine translation.

    Class constructor def __init__(self, vocab, embedding, units, batch):
        vocab is an integer representing the size of the output vocabulary
        embedding is an integer representing the dimensionality of the embedding vector
        units is an integer representing the number
        of hidden units in the RNN cell
        batch is an integer representing the batch size
        Sets the following public instance attributes:
            embedding - a keras Embedding layer that converts
            words from the vocabulary into an embedding vector
            gru - a keras GRU layer with units units
                Should return both the full sequence of outputs
                as well as the last hidden state
                Recurrent weights should be initialized with glorot_uniform
            F - a Dense layer with vocab units
    Public instance method def call(self, x, s_prev, hidden_states):
        x is a tensor of shape (batch, 1) containing the previous word
        in the target sequence as an index of the target vocabulary
        s_prev is a tensor of shape (batch, units) containing the previous
        decoder hidden state
        hidden_states is a tensor of shape
        (batch, input_seq_len, units)containing the outputs of the encoder
        You should use SelfAttention = __import__('1-self_attention')
        .SelfAttention
        You should concatenate the context vector with x in that order
        Returns: y, s
            y is a tensor of shape (batch, vocab) containing the output
            word as a one hot vector in the target vocabulary
            s is a tensor of shape (batch, units) containing the new decoder hidden state
    """

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
        SelfAttention = __import__('1-self_attention').SelfAttention
        x = self.embedding(x)
        attention = SelfAttention(self.units)
        context, _ = attention(s_prev, hidden_states)
        x = tf.cast(x, dtype=tf.float64)
        context = tf.cast(context, dtype=tf.float64)
        context = tf.expand_dims(context, 1)
        cont_x = tf.concat([context, x], axis=2)
        y, hidden = self.gru(cont_x)
        y = y[:, 0, :]
        Y = self.F(y)

        return Y, hidden
