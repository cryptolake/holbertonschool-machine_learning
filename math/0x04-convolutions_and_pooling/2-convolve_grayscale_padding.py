#!/usr/bin/env python3
"""Perform convolution."""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
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

    padding is a tuple of (ph, pw)
        ph is the padding for the height of the image
        pw is the padding for the width of the image
    """
    ph, pw = padding
    kh, kw = kernel.shape
    m, h, w = images.shape
    oh = (h+2*ph-kh)+1
    ow = (w+2*pw-kw)+1
    npad = ((0, 0), (ph, ph), (pw, pw))
    images = np.pad(images, pad_width=npad, mode='constant')
    output = np.zeros((m, oh, ow))
    for i in range(0, oh):
        for j in range(0, ow):
            output[:, i, j] = np.sum(images[:, i:i+kh, j:j+kw] * kernel,
                                     axis=(1, 2))

    return output
