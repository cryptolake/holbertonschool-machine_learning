#!/usr/bin/env python3
"""Bayesian Optimization."""
import numpy as np
from scipy.stats import norm
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
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Acquision function, select the next sample.
        We use Expected improvement, we calculate the expected
        improvement of our potential sample space and select the max/min.

        Returns:
            X_next: the next x to sample (1, 1)
            EI: contains the expected improvement of our potential
            sample space (s, 1)
        """
        # mu: (t, 1)
        mu = self.gp.Y.reshape(-1)
        # mu_s: (s, 1)
        # sigma_s: (s, 1)
        mu_s, sigma_s = self.gp.predict(self.X_s)
        if self.minimize:
            mu_s = -mu_s
            mu = -mu

        # x_m: scalar
        x_m = np.max(mu)
        # z: (s, 1)
        with np.errstate(divide='ignore'):
            imp = mu_s - x_m - self.xsi
            z = imp / sigma_s
            EI = imp * norm.cdf(z) + sigma_s * norm.pdf(z)
            EI[sigma_s == 0.0] = 0.0

        return self.X_s[np.argmax(EI)], EI

    def optimize(self, iterations=100):
        """
        Perform Bayesian optimization.
        
        Optimize a black box function, we use the acquisition function over 
        the bounds to find the next point to sample, then we update update
        our gaussian process with the result and iterate.

        Args:
            iteration: number iteration to use in the algorithms
        
        Returns:
            X_opt: Best x for our optimization goal (maximization/minimization)
            Y_opt: f(X_opt)
        """
        for _ in range(iterations):
            x_s, _ = self.acquisition()
            if x_s in self.gp.X:
                self.gp.Y = self.gp.Y[:-1]
                self.gp.X = self.gp.X[:-1]
                break
            y_s = self.f(x_s)
            self.gp.update(x_s, y_s)
        index = np.argmax(self.gp.Y)
        if self.minimize:
            index = np.argmin(self.gp.Y)
        return self.gp.X[index], self.gp.Y[index]
