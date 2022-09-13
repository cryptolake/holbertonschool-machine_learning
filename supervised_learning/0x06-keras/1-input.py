#!/usr/bin/env python3
"""Build Keras model."""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Build a neural network with the Keras library.

    nx is the number of input features to the network

    layers is a list containing the number of nodes in
    each layer of the network

    activations is a list containing the activation functions
    used for each layer of the network

    lambtha is the L2 regularization parameter

    keep_prob is the probability that a node will be kept for dropout
    Returns: the keras model
    """
    prev = K.Input(shape=(nx,))
    inputs = prev
    l2 = K.regularizers.L2(lambtha)
    for i, layer in enumerate(layers):
        prev = K.layers.Dense(layer, activation=activations[i],
                              kernel_regularizer=l2)(prev)
        if i != len(layers) - 1:
            prev = K.layers.Dropout(1-keep_prob)(prev)

    model = K.Model(inputs=inputs, outputs=prev)
    return model
