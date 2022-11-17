#!/usr/bin/env python3
"""Multivariate gaussian distribution."""
import numpy as np


class MultiNormal:
    """Multivariate gaussian distribution."""

    def __init__(self, data):
        """Initialize."""
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        _, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        self.data = data.T
        self.mean, self.cov = self.mean_conv()

    def mean_conv(self):
        """Get mean and covariance matrix."""
        mean = np.mean(self.data, axis=0, keepdims=True)
        cov = np.matmul((self.data - mean).T, (self.data - mean))\
            / (self.data.shape[0] - 1)
        return mean.T, cov.T
