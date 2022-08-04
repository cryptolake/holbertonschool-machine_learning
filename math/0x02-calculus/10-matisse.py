#!/usr/bin/env python3
"""DERIVATIVE."""


def poly_derivative(poly):
    """
    Find derivative.

    poly is a list of coefficients representing a polynomial
    the index of the list represents the power of x that
    the coefficient belongs to
    Example: if [f(x) = x^3 + 3x +5] , poly is equal to [5, 3, 0, 1]
    If poly is not valid, return None
    If the derivative is 0, return [0]
    Return a new list of coefficients
    representing the derivative of the polynomial
    """
    if len(poly) == 0:
        return None
    derv = []
    for i, x in enumerate(poly):
        derv.append(x)
        if i > 1 and x != 0:
            derv[i] *= i
    derv.pop(0)
    if sum(derv) == 0 or len(derv) == 0:
        return 0
    return derv
