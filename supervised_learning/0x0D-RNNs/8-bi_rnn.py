#!/usr/bin/env python3
"""Forward propagation on a bidirectional RNN."""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Forward propagation on a bidirectional RNN."""
    T, m, _ = X.shape
    _, c_0 = h_0.shape
    _, c_t = h_t.shape
    H_f = np.zeros((T, m, c_0))
    H_b = np.zeros((T, m, c_t))

    for t in range(len(X)):
        H_f[t] = bi_cell.forward(h_0, X[t])
        h_0 = H_f[t]

    for t in range(len(X))[::-1]:
        H_b[t] = bi_cell.backward(h_t, X[t])
        h_t = H_b[t]

    H = np.concatenate((H_f, H_b), axis=2)
    V = bi_cell.output(H)
    return H, V
