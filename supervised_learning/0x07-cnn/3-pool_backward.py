#!/usr/bin/env python3
"""Backward prop pooling."""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs back propagation over a pooling layer of a neural network."""
    kh, kw = kernel_shape
    sh, sw = stride
    m, n_h, n_w, n_c = dA.shape
    dA_prev = np.zeros_like(A_prev)

    for img in range(m):
        for i in range(n_h):
            for j in range(n_w):
                x = i * sh
                y = j * sw
                for k in range(n_c):
                    if mode == 'max':
                        a_prev_s = A_prev[img, x:x+kh, y:y+kw, k]
                        mask = (a_prev_s == np.max(a_prev_s))
                        dA_prev[img, x:x+kh, y:y+kw, k] += (mask
                                                            * dA[img, i, j, k])
                    else:
                        average_dA = dA[img, i, j, k]/(kh*kw)
                        mask = np.ones((kh, kw))
                        dA_prev[img, x:x+kh, y:y+kw, k] += mask*average_dA

    return dA_prev
