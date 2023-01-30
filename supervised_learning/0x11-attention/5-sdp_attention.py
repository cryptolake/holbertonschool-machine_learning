#!/usr/bin/env python3
"""Attention as in the 'attention is all you need' paper."""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Attention as in the 'attention is all you need' paper."""
    dk = K.shape[-1]
    scaled = tf.matmul(Q, K, transpose_b=True)/tf.sqrt(tf.cast(dk, tf.float32))
    if mask is not None:
        mask *= 1e9
        scaled += mask
    else:
        mask = 0
    att_weights = tf.nn.softmax(scaled)
    sdp_att = tf.matmul(att_weights, V)

    return sdp_att, att_weights
