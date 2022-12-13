#!/usr/bin/env python3
"""Initialize RNN GRU cell."""
import numpy as np


class GRUCell:
    """RNN GRU Cell."""

    def __init__(self, i, h, o):
        """Initialize class."""
        self.Wz = np.random.normal(size=(i+h, h))
        self.bz = np.zeros(shape=(1, h))
        self.Wr = np.random.normal(size=(i+h, h))
        self.br = np.zeros(shape=(1, h))
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
        joint = np.concatenate((h_prev, x_t), axis=1)
        z_t = self.sigmoid(joint @ self.Wz + self.bz)
        r_t = self.sigmoid(joint @ self.Wr + self.br)
        h_r = np.concatenate((r_t * h_prev, x_t), axis=1)
        hb_t = self.tanh(h_r @ self.Wh + self.bh)
        h_t = (1-z_t) * h_prev + z_t * hb_t
        y_t = self.softmax(h_t @ self.Wy + self.by)
        return h_t, y_t
