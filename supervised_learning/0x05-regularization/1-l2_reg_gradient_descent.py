#!/usr/bin/env python3
"""L2 Regulazation in Gradient descent."""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """L2 Regulazation in Gradient descent."""
    m = len(Y[0])
    dz = cache['A'+str(L)] - Y
    for i in range(L, 0, -1):
        dw = (1/m) * np.matmul(dz, cache['A'+str(i-1)].T)\
            + (lambtha / m)*weights['W'+str(i)]
        db = (1/m) * np.sum(dz, axis=1, keepdims=True)
        da = np.matmul(weights['W'+str(i)].T, dz)
        dz = da * (1-cache['A'+str(i-1)]**2)

        weights['W'+str(i)] = weights['W'+str(i)]\
            - alpha * dw
        weights['b'+str(i)] = weights['b'+str(i)]\
            - alpha * db
