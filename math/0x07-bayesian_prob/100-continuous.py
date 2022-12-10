#!/usr/bin/env python3
"""
Continuous Posterior.

Given that the prior uses a binomial distribution, its conjugate
the prior we can estimate using a beta distribution.
"""
from scipy import special


def posterior(x, n, p1, p2):
    """Find the range of the probability that\
    the posterior falls between two values."""
    if not isinstance(n, int) or n < 1:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0'
        )
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(p1, float) or p1 < 0 or p1 > 1:
        raise ValueError('p1 must be a float in the range [0, 1]')
    if not isinstance(p2, float) or p2 < 0 or p2 > 1:
        raise ValueError('p2 must be a float in the range [0, 1]')
    if p2 <= p1:
        raise ValueError('p2 must be greater than p1')
    alpha, beta = 1, 1
    return special.btdtr(x+alpha, n-x+beta, p2) -\
        special.btdtr(x+alpha, n-x+beta, p1)
