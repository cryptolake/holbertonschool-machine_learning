#!/usr/bin/env python3
"""
Deep L-layer neural network.

with this we can have a neural network
with an arbitrary number L of layers.
"""
import numpy as np


class DeepNeuralNetwork:
    """A deep neural network."""

    def __init__(self, nx, layers):
        """
        Init the neural network.

        nx: the number of input features
        layers: list representing the number of
        nodes in each layer of the network.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__weights = {}
        for la in range(len(layers)):
            if layers[la] < 1:
                raise TypeError("layers must be a list of positive integers")
            if la == 0:
                self.__weights['W'+str(la+1)] = np.random.randn(layers[la],
                                                                nx) * \
                    np.sqrt(2/nx)
            else:
                self.__weights['W'+str(la+1)] = np.random.randn(layers[la],
                                                                layers[la-1])\
                    * np.sqrt(2/layers[la-1])
            self.__weights['b'+str(la+1)] = np.zeros((layers[la], 1))
        self.__L = len(layers)
        self.__cache = {}

    @property
    def L(self):
        """Getter for L."""
        return self.__L

    @property
    def weights(self):
        """Getter for weights."""
        return self.__weights

    @property
    def cache(self):
        """Getter for cache."""
        return self.__cache

    @staticmethod
    def sigmoid(x):
        """Sigmoid function."""
        return 1.0/(1.0+np.exp(-x))

    def forward_prop(self, X):
        """Forward propagation."""
        self.__cache['A0'] = X
        for i in range(self.__L):
            z = np.dot(self.weights['W'+str(i+1)], self.__cache['A'+str(i)])\
                + self.__weights['b'+str(i+1)]
            A = self.sigmoid(z)
            self.__cache['A'+str(i+1)] = A
        return A, self.__cache

    def cost(self, Y, A):
        """Logistic loss of neural network."""
        loss = np.sum(-(Y*np.log(A)+(1-Y)*np.log(1.0000001 - A)))
        return (1/len(A[0])) * loss

    def evaluate(self, X, Y):
        """Evaluate Model."""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        preds = np.where(A >= 0.5, 1, 0)
        return preds, cost
