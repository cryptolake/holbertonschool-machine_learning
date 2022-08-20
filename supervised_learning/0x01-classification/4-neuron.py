#!/usr/bin/env python3
"""All about a neuron."""
import numpy as np


class Neuron:
    """Single neuron performing binary classification."""

    def __init__(self, nx):
        """
        Initialize the neuron.

        nx: the number of input features to the neuron.

        W: The weights vector for the neuron.
        b: The bias for the neuron.
        A: The activated output of the neuron (prediction).
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for W."""
        return self.__W

    @property
    def b(self):
        """Getter for B."""
        return self.__b

    @property
    def A(self):
        """Getter for A."""
        return self.__A

    @staticmethod
    def sigmoid(x):
        """Sigmoid function."""
        return 1.0/(1.0+np.exp(-x))

    def forward_prop(self, X):
        """Forward propagation."""
        self.__A = self.sigmoid(np.dot(self.W, X)+self.b)
        return self.A

    def cost(self, Y, A):
        """Cost Function."""
        loss = np.sum(-(Y*np.log(A)+(1-Y)*np.log(1.0000001 - A)))
        return (1/len(A[0])) * loss

    def evaluate(self, X, Y):
        """Evaluate neuron."""
        A = self.forward_prop(X)
        pred = np.where(A >= 0.5, 1, 0)
        return pred, self.cost(Y, A)
