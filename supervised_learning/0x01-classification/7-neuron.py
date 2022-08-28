#!/usr/bin/env python3
"""All about a neuron."""
import numpy as np
import matplotlib.pyplot as plt


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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Gradient descent to improve neuron."""
        dz = A - Y
        dw = (1/len(A[0])) * np.matmul(dz, X.transpose())
        db = np.sum((1/len(A[0])) * np.sum(dz, axis=1, keepdims=True))
        self.__W = self.W - alpha * dw
        self.__b = self.b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Train the neuron."""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step > iterations or step < 0:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        for i in range(0, iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.A, alpha)
            costs.append(self.cost(Y, self.A))
            if verbose:
                print(f"Cost after {i} iterations: {self.cost(Y, self.A)}")
            if i != 0 and graph and (i % step) == 0:
                plt.plot(costs)
                plt.xlabel('iteration')
                plt.ylabel('cost')
                plt.title('Training Cost')
                plt.show()

        return self.evaluate(X, Y)
