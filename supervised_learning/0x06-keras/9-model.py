#!/usr/bin/env python3
"""Save and restore keras Model."""
import tensorflow.keras as K


def save_model(network, filename):
    """Save model."""
    K.models.save_model(network, filename)
    return None


def load_model(filename):
    """Load model."""
    model = K.models.load_model(filename)
    return model
