#!/usr/bin/env python3
"""Bayesian Optimization."""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Bayesian Optimization.

    Perform Bayesian optimization on a noiseless 1D gaussian process.
    The objective is to find the x where y is maximum or minimum in the
    least ammount of steps.
    Our assumptions is that our black-box function is smooth and is noiseless.
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """
        Initialize bayesian optimization.

        Args:
            f: black box function to be optimized
            X_init: inputs sampled from f (t, 1)
            Y_init: outputs of X_init sampled f (t, 1)
            bounds: space in which to look for the optimal point
            ac_samples: number of samples to be analyzed during acquisition
            l: length parameter for the GP kernel
            sigma_f: standard deviation given to the output of f
            (function value bounds)
            xsi: exploration exploitation factor
            minimize: determine where to minimize or maximise
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples)
        self.xsi = xsi
        self.minimize = minimize
