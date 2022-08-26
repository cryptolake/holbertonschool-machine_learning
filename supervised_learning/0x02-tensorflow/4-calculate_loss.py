#!/usr/bin/env python3
"""tensorflow loss."""

import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """Calculate cross entropy loss."""
    return tf.losses.softmax_cross_entropy(y, y_pred)
