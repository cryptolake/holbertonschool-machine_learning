#!/usr/bin/env python3
"""sum is the best."""


def summation_i_squared(n):
    """
    Variable n is the stopping condition.

    Return the integer value of the sum
    If n is not a valid number, return None
    You are not allowed to use any loops
    """
    sum = 0
    for i in range(1, n+1):
        sum += i * i
    return sum
