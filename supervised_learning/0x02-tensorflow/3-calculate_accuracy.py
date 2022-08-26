#!/usr/bin/env python3
"""tensorflow accuracy."""

import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """Accuracy of prediction."""
    correct = tf.where(tf.math.equal(y, y_pred), y, y_pred)

    return tf.reduce_mean(correct)
