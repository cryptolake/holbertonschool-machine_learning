#!/usr/bin/env python3
"""Initialize RNN LSTM cell."""
import numpy as np


class LSTMCell:
    """RNN LSTM Cell."""

    def __init__(self, i, h, o):
        """Initialize class."""
        self.Wf = np.random.normal(size=(i+h, h))
        self.bf = np.zeros(shape=(1, h))
        self.Wu = np.random.normal(size=(i+h, h))
        self.bu = np.zeros(shape=(1, h))
        self.Wc = np.random.normal(size=(i+h, h))
        self.bc = np.zeros(shape=(1, h))
        self.Wo = np.random.normal(size=(i+h, h))
        self.bo = np.zeros(shape=(1, h))
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

    def forward(self, h_prev, c_prev, x_t):
        """Perform forward propagation."""
        # Current state
        joint = np.concatenate((h_prev, x_t), axis=1)
        f_t = self.sigmoid(joint @ self.Wf + self.bf)
        u_t = self.sigmoid(joint @ self.Wu + self.bu)
        cb_t = self.tanh(joint @ self.Wc + self.bc)
        c_t = f_t * c_prev + u_t * cb_t
        o_t = self.sigmoid(joint @ self.Wo + self.bo)
        h_t = o_t * self.tanh(c_t)
        y_t = self.softmax(h_t @ self.Wy + self.by)

        return h_t, c_t, y_t
