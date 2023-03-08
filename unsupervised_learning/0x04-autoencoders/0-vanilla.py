#!/usr/bin/env python3
"""
Build a (vanilla) Autoencoder, It will try to reconstruct the input
by first mapping it to  a lower dimension then reconstructing it.
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Build an autoencoder.
    Args:
        input_dims: input dimension which is the same as the output
        hidden_layers: number of nodes in each hidden layer
        latent_dims: dimension of the latent space layer
    Return:
        Model: the autoencoder keras model
    """
    encoder = keras.Sequential(
        [keras.layers.Input((None, input_dims))] +
        [keras.layers.Dense(x, activation='relu') for x in hidden_layers] +
        [keras.layers.Dense(latent_dims)]
    )
    decoder = keras.Sequential(
        [keras.layers.Input((None, latent_dims))] +
        [keras.layers.Dense(x, activation='relu')
         for x in reversed(hidden_layers)] +
        [keras.layers.Dense(input_dims, activation='sigmoid')])
    auto = keras.Sequential([
        keras.layers.Input((None, 784)),
        encoder,
        decoder
    ])
    auto.compile(optimizer='adam', loss='binary_cross_entropy')
    return encoder, decoder, auto
