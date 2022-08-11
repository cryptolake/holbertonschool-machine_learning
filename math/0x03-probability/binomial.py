#!/usr/bin/env python3
"""
Binomial distribution.

 the binomial distribution with parameters n and p
 is the discrete probability distribution of
 the number of successes in a sequence of n
 independent experiments, each asking a yes–no question.
"""


def mean(data):
    """Get mean."""
    return sum(data) / len(data)


def variance(data):
    """Get variance."""
    mn = mean(data)
    return (sum([(x - mn) ** 2 for x in data])
            / len(data))


class Binomial:
    """Class of Binomial."""

    def __init__(self, data=None, n=1, p=0.5):
        """Initialize."""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p > 1 or p <= 0:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            p_value = 1-(variance(data) / mean(data))
            self.n = round(mean(data) / p_value)
            self.p = mean(data) / self.n
