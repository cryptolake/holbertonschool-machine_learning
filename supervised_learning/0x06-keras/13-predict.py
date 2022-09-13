#!/usr/bin/env python3
"""Make prediction."""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Make prediction."""
    prediction = network.predict(data, verbose=verbose)
    return prediction
