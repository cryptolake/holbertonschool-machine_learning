#!/usr/bin/env python3
"""Initialize RNN cell."""
import numpy as np


class RNNCell:
    """RNN Cell."""

    def __init__(self, i, h, o):
        """Initialize class."""
        self.Wh = np.random.normal(size=(i, h))
        self.bh = np.zeros(shape=(1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros(shape=(1, o))

    @staticmethod
    def tanh(x):
        """Sigmoid activation."""
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    @staticmethod
    def softmax(x):
        """Softmax activation."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def forward(self, h_prev, x_t):
        """Perform forward propagation."""
        # Current state
        c_t = self.tanh(x_t @ self.Wh + h_prev + self.bh)
        y_t = self.softmax(c_t @ self.Wy + self.by)
        return c_t, y_t
