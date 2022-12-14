#!/usr/bin/env python3
"""Forward propagation on a deep RNN."""
import numpy as np


def deep_rnn(rnn_cell, X, h_0):
    """Forward propagation on a deep RNN."""
    H = np.expand_dims(h_0, axis=0)
    V = None
    for i, x in enumerate(X):
        h_layers = H[i]
        full_new_l = []
        for j, r_c in enumerate(rnn_cell):
            h_prev_l = h_layers[j]
            h_new_l, y = r_c.forward(h_prev_l, x)
            x = h_new_l
            h_new_l = np.expand_dims(h_new_l, axis=0)
            full_new_l.append(h_new_l)

        full_new_l = np.concatenate(full_new_l, axis=0)
        h_new_layers = np.expand_dims(full_new_l, axis=0)
        H = np.append(H, h_new_layers, axis=0)
        y_add = np.expand_dims(y, axis=0)
        if V is None:
            V = y_add
        else:
            V = np.append(V, y_add, axis=0)
    return H, V
