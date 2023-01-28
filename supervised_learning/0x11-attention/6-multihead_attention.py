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
        multi_head = []
        att_ws = []
        for _ in range(self.h):
            Q_l = self.Wq(Q)
            K_l = self.Wk(K)
            V_l = self.Wv(V)
            sdp_a, att_w = sdp_attention(Q_l, K_l, V_l, mask)
            att_w = tf.expand_dims(att_w, axis=1)
            multi_head.append(sdp_a)
            att_ws.append(att_w)
        multi_head = tf.concat(multi_head, axis=2)
        att_ws = tf.concat(att_ws, axis=1)
        Y = self.linear(multi_head)
        return Y, att_ws
