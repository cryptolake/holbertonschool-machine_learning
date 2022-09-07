#!/usr/bin/env python3
"""Batch Normalization in tensorflow."""
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Batch Normalization Layer in tensorflow.

    prev is the activated output of the previous layer

    n is the number of nodes in the layer to be created

    activation is the activation function that should
    be used on the output of the layer

    Returns: a tensor of the activated output for the layer
    """
    mean = tf.math.reduce_mean(prev)
    var = tf.math.reduce_variance(prev)
    gamma = tf.Variable(1., trainable=True)
    beta = tf.Variable(0., trainable=True)
    normalized = tf.nn.batch_normalization(prev, mean, var, beta, gamma, 1e-8)
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(n, activation, kernel_initializer=init)
    return layer(normalized)
