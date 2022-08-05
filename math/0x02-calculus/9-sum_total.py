#!/usr/bin/env python3
"""sum is the best."""


def pow(n, p):
    """Power."""
    if p == 0:
        return 1
    return n * pow(n, p-1)


def summation_i_squared(n):
    """
    Variable n is the stopping condition.

    Return the integer value of the sum
    If n is not a valid number, return None
    You are not allowed to use any loops
    """
    if n < 1 or type(n) is not int:
        return None
    return int((2 * pow(n, 3) + 3 * pow(n, 2) + n)/6)
