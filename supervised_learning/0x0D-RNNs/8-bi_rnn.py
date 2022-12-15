#!/usr/bin/env python3
"""Forward propagation on a bidirectional RNN."""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Forward propagation on a bidirectional RNN."""
    H = None
    V = None
    for x_f, x_b in zip(X, np.flip(X, axis=0)):
        h_0 = bi_cell.forward(h_0, x_f)
        h_t = bi_cell.backward(h_t, x_b)
        H_n = np.concatenate((h_0, h_t), axis=-1)
        Y = bi_cell.output(H_n)
        Y = np.expand_dims(Y, axis=0)
        H_n = np.expand_dims(H_n, axis=0)
        if H is None:
            H = H_n
        else:
            H = np.append(H, H_n, axis=0)
        if V is None:
            V = Y
        else:
            V = np.append(V, Y, axis=0)
    return H, V
