#!/usr/bin/env python3

"""
Normal distribution.

is a type of continuous probability
distribution for a real-valued random variable.
"""



class Normal:
    """Class of Normal."""


    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize."""
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.mean = mean
            self.stddev = stddev
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = sum(data) / len(data)
            self.stddev = (sum([(x - self.mean)**2 for x in data])
                           / len(data)) ** 1/2
