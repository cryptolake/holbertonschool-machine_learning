#!/usr/bin/env python3
"""Tensorflow layers."""

import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """Create tensorflow layer."""
    layer = tf.keras.layers.Dense(n, activation=activation, name="layer")
    return layer(prev)
