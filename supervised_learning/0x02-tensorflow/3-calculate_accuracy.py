#!/usr/bin/env python3
"""tensorflow accuracy."""

import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """Accuracy of prediction."""
    y = tf.argmax(y, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)
    correct = tf.equal(y, y_pred)
    return tf.reduce_mean(tf.cast(correct, dtype='float'))
