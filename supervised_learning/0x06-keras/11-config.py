#!/usr/bin/env python3
"""Save and restore keras Model weights."""
import tensorflow.keras as K


def save_config(network, filename):
    """Save model."""
    with open(filename, 'w') as modeljson:
        modeljson.write(network.to_json())
    return None


def load_config(filename):
    """Load model."""
    with open(filename, 'r') as modeljson:
        model = K.models.model_from_json(modeljson.read())
    return model
