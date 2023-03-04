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
        norm = (X1**2) + (X2**2).T - 2*(X1 @ X2.T)
        rbf = self.sigma_f**2 * np.exp(-0.5/self.l**2 * norm)
        return rbf
