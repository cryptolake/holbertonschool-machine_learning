#!/usr/bin/env python3
"""GMM using sklearn."""
import sklearn.mixture


def gmm(X, k):
    """GMM using sklearn."""
    gm = sklearn.mixture.GaussianMixture(n_components=k)
    labels = gm.fit_predict(X)
    bic = gm.bic(X)
    return gm.weights_, gm.means_, gm.covariances_, labels, bic
