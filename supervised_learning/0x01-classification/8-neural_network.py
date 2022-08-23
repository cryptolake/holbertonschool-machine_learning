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
        self.W1 = np.random.normal(size=(nodes, nx))
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0
