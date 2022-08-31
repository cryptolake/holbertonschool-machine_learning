#!/usr/bin/env python3
"""
Deep L-layer neural network.

with this we can have a neural network
with an arbitrary number L of layers.
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


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
                                                                nx)\
                    * np.sqrt(2/nx)
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
        for i in range(self.L):
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

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Gradient descent as back propagation."""
        m = len(Y[0])
        da = (-Y)/cache['A'+str(self.L)]\
            + (1-Y)/(1-cache['A'+str(self.L)])
        for i in range(self.L, 0, -1):
            dz = da * (cache['A'+str(i)] *
                       (1-cache['A'+str(i)]))
            dw = (1/m) * np.matmul(dz, cache['A'+str(i-1)].transpose())
            db = (1/m) * np.sum(dz, axis=1, keepdims=True)
            da = np.matmul(self.__weights['W'+str(i)].transpose(), dz)
            self.__weights['W'+str(i)] = self.__weights['W'+str(i)]\
                - alpha * dw
            self.__weights['b'+str(i)] = self.__weights['b'+str(i)]\
                - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Train the neural network."""
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
            self.gradient_descent(Y, self.cache, alpha)
            costs.append(self.cost(Y, self.cache['A'+str(self.L)]))
            if verbose:
                print(f"Cost after {i} iterations: {costs[-1]}")
            if i != 0 and graph and (i % step) == 0:
                plt.plot(costs)
                plt.xlabel('iteration')
                plt.ylabel('cost')
                plt.title('Training Cost')
                plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """Save as pickkl file."""
        if filename[-4:] != '.pkl':
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            f.close()

    @staticmethod
    def load(filename):
        """Load DeepNeuralNetwork from pickle file."""
        with open(filename, 'rb') as f:
            try:
                model = pickle.load(f)
                f.close()
            except Exception:
                return None
        return model
