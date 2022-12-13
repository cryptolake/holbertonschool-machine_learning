#!/usr/bin/env python3
"""Forward propagation on a simple RNN."""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Forward propagation on a simple RNN."""
    h_prev = h_0
    H = np.expand_dims(h_prev, axis=0)
    V = None
    for x in X:
        h_prev, y = rnn_cell.forward(h_prev, x)
        h_add = np.expand_dims(h_prev, axis=0)
        y_add = np.expand_dims(y, axis=0)
        if V is None:
            V = y_add
        else:
            V = np.append(V, y_add, axis=0)
        H = np.append(H, h_add, axis=0)
    return H, V
