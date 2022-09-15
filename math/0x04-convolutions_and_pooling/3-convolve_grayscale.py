#!/usr/bin/env python3
"""Perform convolution."""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Perform Valid Discreet convolution.

    images is a numpy.ndarray with shape (m, h, w) containing
    multiple grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images

    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel
    for the convolution
        kh is the height of the kernel
        kw is the width of the kernel

    padding is either a tuple of (ph, pw), 'same', or 'valid'
        if 'same', performs a same convolution
        if 'valid', performs a valid convolution
        if a tuple:
        ph is the padding for the height of the image
        pw is the padding for the width of the image
    """
    kh, kw = kernel.shape
    sh, sw = stride
    if type(padding) is tuple:
        ph, pw = padding
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph = int(np.ceil((h*(sh-1)+kh-1)/2))
        pw = int(np.ceil((w*(sw-1)+kw-1)/2))
    m, h, w = images.shape
    oh = int((h+2*ph-kh)/sh+1)
    ow = int((w+2*pw-kw)/sw+1)
    npad = ((0, 0), (ph, ph), (pw, pw))
    images = np.pad(images, pad_width=npad, mode='constant')
    output = np.zeros((m, oh, ow))
    for i in range(0, oh):
        x = i * sh
        for j in range(0, ow):
            y = j * sw
            output[:, i, j] = np.sum(images[:, x:x+kh, y:y+kw] * kernel,
                                     axis=(1, 2))

    return output
