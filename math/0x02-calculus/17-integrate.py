#!/usr/bin/env python3
"""Poly integration."""


def poly_integral(poly, C=0):
    """Polynomial Integration."""
    if type(poly) is not list or type(C) is not int or len(poly) == 0:
        return None
    if len(poly) == 1 and poly[0] == 0:
        poly[0] = C
        return poly
    poly.insert(0, C)
    integ = []
    for i, x in enumerate(poly):
        integ.append(x)
        if x != 0 and i != 0:
            integ[i] = x / i
            if integ[i] == int(integ[i]):
                integ[i] = int(integ[i])
    return integ
