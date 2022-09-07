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
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(n, kernel_initializer=init)
    z = layer(prev)
    gamma = tf.Variable(1., trainable=True)
    beta = tf.Variable(0., trainable=True)
    mean = tf.math.reduce_mean(z, axis=0)
    var = tf.math.reduce_variance(z, axis=0)
    epsilon = 1e-8
    normalized = tf.nn.batch_normalization(z, mean, var, beta, gamma, epsilon)
    return activation(normalized)
