#!/usr/bin/env python3
"""Evaluate model."""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Test model."""
    result = network.evaluate(data, labels, verbose=verbose)
    return result
