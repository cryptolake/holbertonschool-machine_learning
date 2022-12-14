#!/usr/bin/env python3
"""Initialize RNN Bidrectional cell."""
import numpy as np


class BidirectionalCell:
    """RNN Bidrectional Cell."""

    def __init__(self, i, h, o):
        """Initialize class."""
        self.Whf = np.random.normal(size=(i+h, h))
        self.bhf = np.zeros(shape=(1, h))
        self.Whb = np.random.normal(size=(i+h, h))
        self.bhb = np.zeros(shape=(1, h))
        self.Wy = np.random.normal(size=(h+h, o))
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
        """Perform forward in the forward direction propagation."""
        joint = np.concatenate((h_prev, x_t), axis=1)
        zf = joint @ self.Whf + self.bhf
        hf_t = self.tanh(zf)
        return hf_t

    def backward(self, h_next, x_t):
        """Perform forward in the backward direction propagation."""
        joint = np.concatenate((h_next, x_t), axis=1)
        zb = joint @ self.Whb + self.bhb
        hb_t = self.tanh(zb)
        return hb_t
