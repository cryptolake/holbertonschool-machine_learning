#!/usr/bin/env python3
"""Create masks for the transformer training."""
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """Create masks for the transformer training."""
    l_out = tf.shape(target)[1]

    padding_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]

    look_ahead = 1 - tf.linalg.band_part(tf.ones((l_out, l_out)), -1, 0)

    target_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    target_mask = target_mask[:, tf.newaxis, tf.newaxis, :]

    combined_mask = tf.maximum(target_mask, look_ahead)

    return padding_mask, combined_mask, padding_mask
