#!/usr/bin/env python3
"""Perform pooling."""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Perform pooling.

    images is a numpy.ndarray with shape (m, h, w, c)
    containing multiple images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image

    kernel_shape is a tuple of (kh, kw) containing the kernel
    shape for the pooling
        kh is the height of the kernel
        kw is the width of the kernel

    stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image

    mode indicates the type of pooling
        max indicates max pooling
        avg indicates average pooling
    """
    kh, kw = kernel_shape
    sh, sw = stride
    m, h, w, c = images.shape
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
            output[:, i, j, :] = func(images[:, x:x+kh, y:y+kw], axis=(1, 2))

    return output
