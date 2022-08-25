#!/usr/bin/env python3
"""Tensorflow layers."""

import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """Create tensorflow layer."""
    weight = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(n, activation=activation, name="layer",
                                  kernel_initializer=weight)
    return layer(prev)
