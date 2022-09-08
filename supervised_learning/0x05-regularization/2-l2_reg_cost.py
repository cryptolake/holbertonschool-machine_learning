#!/usr/bin/env python3
"""L2 Regulazation cost in tensorflow."""
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """L2 regulazation cost in tensorflow."""
    reg = tf.keras.regularizers.L2()
    return reg(cost)
