#!/usr/bin/env python3
"""from one neuron to a network."""
import numpy as np


class NeuralNetwork:
    """A Neural Network from a neuron."""

    def __init__(self, nx, nodes):
        """
        Initialize The Neural Network.

        nx: the number of input features to the neuron.
        nodes: the number of nodes found in the hidden layer
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for W1."""
        return self.__W1

    @property
    def b1(self):
        """Getter b1."""
        return self.__b1

    @property
    def A1(self):
        """Getter A1."""
        return self.__A1

    @property
    def W2(self):
        """Getter W2."""
        return self.__W2

    @property
    def b2(self):
        """Getter b2."""
        return self.__b2

    @property
    def A2(self):
        """Getter A2."""
        return self.__A2

    @staticmethod
    def sigmoid(x):
        """Sigmoid function."""
        return 1.0/(1.0+np.exp(-x))

    def forward_prop(self, X):
        """Forward propagation."""
        self.__A1 = self.sigmoid(np.dot(self.W1, X)+self.b1)
        self.__A2 = self.sigmoid(np.dot(self.W2,  self.A1) + self.b2)
        return self.A1, self.A2
