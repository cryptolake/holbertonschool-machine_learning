#!/usr/bin/env python3
"""Initialize RNN cell."""
import numpy as np


class RNNCell:
    """RNN Cell."""

    def __init__(self, i, h, o):
        """Initialize class."""
        self.Wh = np.random.normal(size=(i+h, h))
        self.bh = np.zeros(shape=(1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros(shape=(1, o))

    @staticmethod
    def sigmoid(z):
        """Sigmoid function."""
        return 1.0/(1.0+np.exp(-z))

    @staticmethod
    def tanh(z):
        """Tanh activation function."""
        return ((np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z)))

    @staticmethod
    def softmax(z):
        """Softmax function."""
        ex = np.exp(z)
        return ex / np.sum(ex, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """Perform forward propagation."""
        # Current state
        z = np.concatenate((h_prev, x_t), axis=1) @ self.Wh + self.bh
        c_t = self.tanh(z)
        y_t = self.softmax(c_t @ self.Wy + self.by)
        return c_t, y_t
