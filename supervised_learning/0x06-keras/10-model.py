#!/usr/bin/env python3
"""Save and restore keras Model weights."""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """Save model."""
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """Load model."""
    network.load_weights(filename)
    return None
