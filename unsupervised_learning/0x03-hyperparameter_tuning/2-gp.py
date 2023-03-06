#!/usr/bin/env python3
"""Gaussian Process."""
import numpy as np


class GaussianProcess:
    """
    A Gaussian process.

    This class repressents a gaussian process in which
    we initialize with X_t a list of t sameples (x) with
    their prespective Y_t.

    We then produce kerneel K covariance matrix of all x
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initialize Gaussian Process.

        X_init is a numpy.ndarray of shape (t, 1)
        representing the inputs already sampled with the black-box function

        Y_init is a numpy.ndarray of shape (t, 1) representing the outputs
        of the black-box function for each input in X_init

        t is the number of initial samples
        l is the length parameter for the kernel
        sigma_f is the standard deviation given to the output
        of the black-box function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f

        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Caculate the covariance kernel using RBF (Radial Basis Function).

        Args:
            X1: numpy.ndarray of shape (m, 1)
            X2: numpy.ndarray of shape (n, 1)

        Returns:
            Covariance kernel (m, n)
        """
        # (a-b)**2 = a**2 + b**2 - 2ab
        norm = (X1**2) + (X2**2).T - 2 * (X1 @ X2.T)
        rbf = self.sigma_f**2 * np.exp(-0.5 / self.l**2 * norm)
        return rbf

    def predict(self, X_s):
        """
        Predict the mean and standard deviation in the Gaussian process.

        Args:
            X_s: Points to calculate mu and sigma for, (s,)

        Returns:
            mu: Mean of each data point (s,)
            sigma: standard deviation of each data point (s,)
        """
        k = self.K
        k_s = self.kernel(self.X, X_s)
        k_ss = self.kernel(X_s, X_s)

        mu = k_s.T @ np.linalg.inv(k) @ self.Y
        sigma = np.diag(k_ss) - np.diag(k_s.T @ np.linalg.inv(k) @ k_s)

        return mu.reshape(-1), sigma

    def update(self, X_new, Y_new):
        """
        Update Gaussian process.

        Args:
            X_new: x new data point to add (1,)
            Y_new: y new data point to add (1,)
        """
        self.X = np.append(self.X, X_new).reshape(-1, 1)
        self.Y = np.append(self.Y, Y_new).reshape(-1, 1)
        self.K = self.kernel(self.X, self.X)
