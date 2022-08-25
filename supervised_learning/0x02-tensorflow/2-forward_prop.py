#!/usr/bin/env python3
"""Tensorflow forward prop."""

import tensorflow.compat.v1 as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Forward propagation."""
    xi = None
    for layer, activation in zip(layer_sizes, activations):
        xi = create_layer(x, layer, activation)
        x = xi

    return xi
