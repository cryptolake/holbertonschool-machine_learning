#!/usr/bin/env python3
"""Create masks for the transformer training."""
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """Create masks for the transformer training."""
    batch_size, l_in = tf.shape(inputs)
    _, l_out = tf.shape(target)

    padding_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    padding_mask = tf.reshape(padding_mask, shape=(batch_size, 1, 1, l_in))

    combined_mask = 1 - tf.linalg.band_part(tf.ones((batch_size,
                                                     l_out, l_out)), -1, 0)
    combined_mask = tf.reshape(combined_mask, shape=(batch_size, 1,
                                                     l_out, l_out))

    return padding_mask, combined_mask, padding_mask
