#!/usr/bin/env python3
"""Attention as in the 'attention is all you need' paper."""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Attention as in the 'attention is all you need' paper."""
    qk_matmul = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(K.shape[-1], tf.float32)
    scaled = qk_matmul / tf.math.sqrt(dk)
    if mask is not None:
        scaled += (mask * -1e9)

    att_weights = tf.nn.softmax(scaled, axis=-1)
    sdp_att = tf.matmul(att_weights, V)

    return sdp_att, att_weights
