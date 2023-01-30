#!/usr/bin/env python3
"""Multihead Attention as in the 'attention is all you need' paper."""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi head attention as implemented in the paper."""

    def __init__(self, dm, h):
        """Initialize instance."""
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask):
        """Functionality of the layer."""
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        batch, _, _ = Q.shape

        # Split logic from this tutorial:
        # https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853
        def split(tensor):
            """
            Split tensor.

            from (batch, seq, emb) to (batch, head, seq, depth).
            """
            tensor = tf.reshape(tensor, (batch, -1, self.h, self.depth))
            return tf.transpose(tensor, perm=[0, 2, 1, 3])
        Q, K, V = split(Q), split(K), split(V)
        sdp_a, att_w = sdp_attention(Q, K, V, mask)
        swap = tf.transpose(sdp_a, perm=[0, 2, 1, 3])
        merged = tf.reshape(swap, (batch, -1, self.dm))
        Y = self.linear(merged)
        return Y, att_w
