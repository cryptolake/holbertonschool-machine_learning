#!/usr/bin/env python3
"""sum is the best."""


def summation_i_squared(n, i=0):
    """
    Variable n is the stopping condition.

    Return the integer value of the sum
    If n is not a valid number, return None
    You are not allowed to use any loops
    """
    if n == 0:
        return 0
    i += 1
    return i*i + summation_i_squared(n-1, i)
