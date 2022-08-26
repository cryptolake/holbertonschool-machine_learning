#!/usr/bin/env python3
"""Gradient descent in tensorflow."""

import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """Create training operation."""
    train = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    grads = train.compute_gradients(loss)
    return train.apply_gradients(grads)
