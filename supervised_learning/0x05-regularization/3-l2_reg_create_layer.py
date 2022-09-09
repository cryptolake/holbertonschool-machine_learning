#!/usr/bin/env python3
"""L2 Regulazation cost layer in tensorflow."""
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Create layer with L2 Regulazation."""
    l2 = tf.keras.regularizers.L2(lambtha)
    layer = tf.layers.Dense(n, activation=activation,
                                  kernel_regularizer=l2)
    return layer(prev)
