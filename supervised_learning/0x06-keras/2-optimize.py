#!/usr/bin/env python3
"""Optimize and compile Keras model."""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Set up Adam optimization for a keras model with\
    categorical crossentropy loss and accuracy metrics.

    network is the model to optimize
    alpha is the learning rate
    beta1 is the first Adam optimization parameter
    beta2 is the second Adam optimization parameter
    Returns: None
    """
    optimizer = K.optimizers.Adam(alpha, beta1, beta2)
    network.compile(optimizer=optimizer,
                    loss=K.losses.CategoricalCrossentropy(),
                    metrics='accuracy')
