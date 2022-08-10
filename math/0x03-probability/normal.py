#!/usr/bin/env python3

"""
Normal distribution.

is a type of continuous probability
distribution for a real-valued random variable.
"""

PI = 3.1415926536
E = 2.7182818285


def erf(x):
    """Error function provided."""
    return 2/(PI ** (1/2)) * (
           x - (x ** 3) / 3 + (x ** 5) / 10 - (x ** 7) / 42 +
           (x ** 9) / 216)


class Normal:
    """Class of Normal."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize."""
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = sum(data) / len(data)
            self.stddev = (sum([(x - self.mean) ** 2 for x in data])
                           / len(data)) ** (1/2)

    def z_score(self, x):
        """Get z score of x."""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Get x from z score."""
        return (self.stddev * z) + self.mean

    def pdf(self, x):
        """Probability density function."""
        return 1/((self.stddev*(2*PI)**(1/2)) **
                  -1/2*((x-self.mean)/self.stddev))
