#!/usr/bin/env python3
"""RMSProp optimization algorithm."""

import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Implement RMSProp with tensorflow."""
    train = tf.train.RMSPropOptimizer(alpha, epsilon=epsilon, decay=beta2)
    grads = train.compute_gradients(loss)
    return train.apply_gradients(grads)
