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
        self.weights = {}
        for la in range(len(layers)):
            if layers[la] < 1:
                raise TypeError("layers must be a list of positive integers")
            if la == 0:
                self.weights[f'W{la+1}'] = np.random.randn(layers[la],
                                                           layers[la])*\
                    np.sqrt(2/layers[la])
            else:
                self.weights[f'W{la+1}'] = np.random.randn(layers[la],
                                                           layers[la-1])*\
                    np.sqrt(2/layers[la-1])
            self.weights[f'b{la+1}'] = np.zeros((layers[la], 1))
        self.L = len(layers)
        self.cache = {}
