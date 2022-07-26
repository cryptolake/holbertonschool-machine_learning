#!/usr/bin/env python3
"""Implement moving average."""


def moving_average(data, beta):
    """
    Calculate the weighted moving average of a data set.

    data: the list of data to calculate the moving average of
    beta: the weight used for the moving average

    Returns: a list containing the moving averages of data
    """
    v = 0.0
    amw = []
    for i, p in enumerate(data):
        v = beta * v + (1-beta)*p
        amw.append(v/(1-beta**(i+1)))
    return amw
