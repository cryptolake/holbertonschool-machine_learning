#!/usr/bin/env python3
"""Likelihood."""
import numpy as np


def comb(n, k):
    """Calculate binomial coefficent."""
    fact = np.math.factorial
    return fact(n)/(fact(k)*fact(n-k))


def likelihood(x, n, p):
    """Likelihood."""
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError("x must be an integer that\
                is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(p) is not np.ndarray or len(p.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    for v in p:
        if v < 0 or v > 1:
            raise ValueError("All values in P must be in the range [0, 1]")

    return comb(n, x) * (p**x) * ((1.0-p)**(n-x))
