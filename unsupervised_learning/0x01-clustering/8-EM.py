#!/usr/bin/env python3
"""Perform complete EM for GMM."""

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Perform complete EM for GMM."""
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) is not float:
        return None, None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None, None
    pi, m, S = initialize(X, k)
    if pi is None:
        return None, None, None, None, None
    ll = 0
    for i in range(iterations):
        g, nl = expectation(X, pi, m, S)
        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {:.5f}"
                  .format(i, nl))
        if abs(ll-nl) <= tol:
            g, nl = expectation(X, pi, m, S)
            if verbose:
                print("Log Likelihood after {} iterations: {:.5f}"
                      .format(i, nl))
            return pi, m, S, g, nl
        pi, m, S = maximization(X, g)
        ll = nl

    g, nl = expectation(X, pi, m, S)
    if verbose:
        print("Log Likelihood after {} iterations: {:.5f}"
              .format(iterations, nl))
    return pi, m, S, g, nl
