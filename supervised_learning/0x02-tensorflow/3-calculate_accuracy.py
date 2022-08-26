#!/usr/bin/env python3
"""tensorflow accuracy."""

import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """Accuracy of prediction."""
    correct = tf.math.equal(y, y_pred)

    return tf.reduce_mean(tf.cast(correct, "float"))
