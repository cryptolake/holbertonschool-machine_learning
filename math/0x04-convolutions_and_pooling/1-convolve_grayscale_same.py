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
    m, h, w = images.shape
    oh = (h-kh)+1
    ow = (w-kw)+1
    ph = abs((oh-h)//2)
    pw = abs((ow-w)//2)
    oh += 2*ph
    ow += 2*pw
    npad = ((0, 0), (ph, ph), (pw, pw))
    images = np.pad(images, npad, mode='constant', constant_values=0)
    kernel = np.repeat(kernel[np.newaxis, :, :], m, axis=0)
    output = np.zeros((m, oh, ow))
    for i in range(0, oh):
        for j in range(0, ow):
            output[:, i, j] = np.sum(np.sum(images[:, i:i+kw, j:j+kh] * kernel,
                                            axis=2), axis=1)

    return output
