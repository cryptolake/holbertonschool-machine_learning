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


class Poisson:
    """Class of Poisson."""

    def __init__(self, data=None, lambtha=1):
        """Initialize."""
        if data is None:
            if float(lambtha) <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                TypeError('data must be a list')
            elif len(data) < 2:
                ValueError('data must contain multiple values')
            self.lambtha = float(sum(data) / len(data))
