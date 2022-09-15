#!/usr/bin/env python3
"""Perform Same Discreet convolution."""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Perform same Discreet convolution.

    images is a numpy.ndarray with shape (m, h, w) containing
    multiple grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images

    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel
    for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    """
    kh, kw = kernel.shape
    _, h, w = images.shape
    output = np.zeros(images.shape)
    ph = int(np.ceil((kh-1)/2))
    pw = int(np.ceil((kw-1)/2))
    npad = ((0, 0), (ph, ph), (pw, pw))
    images = np.pad(images, pad_width=npad, mode='constant')
    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.sum(images[:, i:i+kh, j:j+kw] * kernel,
                                     axis=(1, 2))

    return output
