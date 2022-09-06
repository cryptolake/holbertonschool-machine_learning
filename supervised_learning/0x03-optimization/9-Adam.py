#!/usr/bin/env python3
"""Adam optimization."""

import tensorflow.compat.v1 as tf


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Implement Adam from scratch.

    alpha is the learning rate
    beta1 is the weight used for the first moment
    beta2 is the weight used for the second moment
    epsilon is a small number to avoid division by zero
    var is a numpy.ndarray containing the variable to be updated
    grad is a numpy.ndarray containing the gradient of var
    v is the previous first moment of var
    s is the previous second moment of var
    t is the time step used for bias correction
    Returns: the updated variable, the new first moment,
    and the new second moment, respectively
    """
    vn = beta1 * v + (1-beta1)*grad
    sn = beta2 * s + (1-beta2)*(grad**2)

    vnc = vn / (1-(beta1**t))
    snc = sn / (1-(beta2**t))

    new_var = var - alpha * (vnc/(snc**(1/2)+epsilon))
    return new_var, vn, sn
