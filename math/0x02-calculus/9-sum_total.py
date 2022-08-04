#!/usr/bin/env python3
"""sum is the best."""


def recur_sum(n, i):
    """Recursive func."""
    if n == 0:
        return 0
    i += 1
    return i*i + recur_sum(n-1, i)


def summation_i_squared(n):
    """
    Variable n is the stopping condition.

    Return the integer value of the sum
    If n is not a valid number, return None
    You are not allowed to use any loops
    """
    if n < 1 or type(n) is not int or type(n) is not float:
        return None
    return recur_sum(int(n), 0)
