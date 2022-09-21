#!/usr/bin/env python3
"""Pooling Forward propagation."""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Perform forward propagation.

    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    kernel_shape is a tuple of (kh, kw) containing the
    size of the kernel for the pooling
        kh is the kernel height
        kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the pooling
        sh is the stride for the height
        sw is the stride for the width
    mode is a string containing either max or avg, indicating whether
    to perform maximum or average pooling, respectively
    """
    kh, kw = kernel_shape
    sh, sw = stride
    m, h, w, c = A_prev.shape
    if mode == 'max':
        func = np.max
    elif mode == 'avg':
        func = np.average

    oh = int((h-kh)/sh+1)
    ow = int((w-kw)/sw+1)
    output = np.zeros((m, oh, ow, c))
    for i in range(0, oh):
        x = i * sh
        for j in range(0, ow):
            y = j * sw
            output[:, i, j, :] = func(A_prev[:, x:x+kh, y:y+kw], axis=(1, 2))

    return output
