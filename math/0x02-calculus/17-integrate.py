#!/usr/bin/env python3
"""Poly integration."""


def poly_integral(poly, C=0):
    """Polynomial Integration."""
    poly.insert(0, C)
    integ = []
    for i, x in enumerate(poly):
        integ.append(x)
        if x != 0:
            integ[i] = x / i
    return integ
