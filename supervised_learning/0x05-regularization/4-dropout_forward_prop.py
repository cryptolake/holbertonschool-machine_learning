#!/usr/bin/env python3
"""L2 Regulazation cost layer in tensorflow."""
import numpy as np


def softmax(z):
    """Softmax function."""
    ex = np.exp(z)
    return ex / ex.sum(axis=0, keepdims=True)


def tanh(z):
    """Tanh activation function."""
    return ((np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z)))


def dropout_forward_prop(X, weights, L, keep_prob):
    """Create forward pro with dropout."""
    cache = {}
    cache['A0'] = X
    for i in range(1, L+1):
        z = np.matmul(weights['W'+str(i)], cache['A'+str(i-1)])\
            + weights['b'+str(i)]
        if i != L:
            A = tanh(z)
            drop = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            drop = drop.astype(int)
            A *= drop
            A /= keep_prob
            cache['D'+str(i)] = drop
        else:
            A = softmax(z)
        cache['A'+str(i)] = A
    return cache
