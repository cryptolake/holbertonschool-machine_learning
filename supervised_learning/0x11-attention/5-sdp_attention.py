#!/usr/bin/env python3
"""Attention as in the 'attention is all you need' paper."""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Attention as in the 'attention is all you need' paper."""
    batches, dk, _ = K.shape
    weights = []
    for batch in range(batches):
        res = tf.expand_dims(Q[batch] @ tf.transpose(K[batch]), 0)
        weights.append(res)
    if mask is not None:
        mask *= 1e9
    else:
        mask = 0

    scaled = tf.concat(weights, 0)/tf.sqrt(float(dk)) + mask
    att_weights = tf.nn.softmax(scaled)

    att = []
    for batch in range(batches):
        res = tf.expand_dims(att_weights[batch] @ V[batch], 0)
        att.append(res)
    sdp_att = tf.concat(att, 0)
    return sdp_att, att_weights
