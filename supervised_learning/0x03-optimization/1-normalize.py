#!/usr/bin/env python3
"""Normalization."""
import numpy as np


def normalize(x, m, s):
    """
    Normalize (standardizes) a matrix.

    X is the numpy.ndarray of shape (d, nx) to normalize:
        d is the number of data points
        nx is the number of features

    m is a numpy.ndarray of shape (nx,)
    that contains the mean of all features of X

    s is a numpy.ndarray of shape (nx,)
    that contains the standard deviation of all features of X

    Returns: The normalized X matrix
    """
    return (x-m)/np.sqrt(s)
