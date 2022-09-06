#!/usr/bin/env python3
"""Gradient descent with momentum."""

import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Training with momentum gradient descent.

    loss is the loss of the network
    alpha is the learning rate
    beta1 is the momentum weight
    """
    train = tf.train.MomentumOptimizer(alpha, beta1)
    grads = train.compute_gradients(loss)
    return train.apply_gradients(grads)
