#!/usr/bin/env python3
"""
Poisson distribution.

the Poisson distribution is a discrete probability distribution
that expresses the probability of a given number
of events occurring in a fixed interval of time
or space if these events occur with a known constant mean rate
and independently of the time since the last event.
"""

E = 2.7182818285


def fact(n):
    """Factorial."""
    if n == 1:
        return 1
    return n * fact(n - 1)


class Poisson:
    """Class of Poisson."""

    def __init__(self, data=None, lambtha=1):
        """Initialize."""
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Probability mass function."""
        if type(k) is not int:
            k = int(k)
        if k < 0:
            k = 0
        pmf = (pow(self.lambtha, k) * pow(E, -self.lambtha)) / fact(k)
        return pmf
