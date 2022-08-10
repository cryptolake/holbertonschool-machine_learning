#!/usr/bin/env python3
"""
Exponential distribution.

the exponential distribution is the probability
distribution of the time between events
in a Poisson point process, i.e., a process
in which events occur continuously and
independently at a constant average rate.
"""

E = 2.7182818285


class Exponential:
    """Class of Exponential."""

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
            self.lambtha = float(sum(data) / len(data)) ** -1

    def pdf(self, x):
        """Probability density function."""
        if x < 0:
            return 0
        pdf = self.lambtha * (E ** -(self.lambtha * x))
        return pdf
