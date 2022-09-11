#!/usr/bin/env python3
"""Dropout in NN."""
import numpy as np

def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Gradient descent with dropout."""
    m = len(Y[0])
    dz = cache['A'+str(L)] - Y
    for i in range(L, 0, -1):
        dw = (1/m) * np.matmul(dz, cache['A'+str(i-1)].T)
        db = (1/m) * np.sum(dz, axis=1, keepdims=True)
        da = np.matmul(weights['W'+str(i)].T, dz)
        if i != 1:
            drop = cache['D'+str(i-1)]
            da *= drop
            da /= keep_prob
        dz = da * (1-cache['A'+str(i-1)]**2)

        weights['W'+str(i)] = weights['W'+str(i)]\
            - alpha * dw
        weights['b'+str(i)] = weights['b'+str(i)]\
            - alpha * db
