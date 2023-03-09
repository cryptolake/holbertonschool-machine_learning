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
        hidden_layers: list number of nodes in each hidden layer
        latent_dims: dimension of the latent space layer
    Return:
        Model: the autoencoder keras model
    """
    # encoder = keras.Sequential(
    #     [keras.Input((input_dims,))] +
    #     [keras.layers.Dense(x, activation='relu') for x in hidden_layers] +
    #     [keras.layers.Dense(latent_dims)]
    # )
    # decoder = keras.Sequential(
    #     [keras.layers.Input((latent_dims,))] +
    #     [keras.layers.Dense(x, activation='relu')
    #      for x in reversed(hidden_layers)] +
    #     [keras.layers.Dense(input_dims, activation='sigmoid')])
    # auto = keras.Sequential([
    #     keras.Input((input_dims,)),
    #     encoder,
    #     decoder
    # ])

    # Checker won't accept a keras Sequential model

    # The encoder
    X = keras.Input((input_dims,))
    en = X
    for nl in hidden_layers:
        en = keras.layers.Dense(nl, activation='relu')(en)
    en_final = keras.layers.Dense(latent_dims, activation='relu')(en)
    encoder = keras.Model(X, en_final)

    # The Decoder
    de_X = keras.Input((latent_dims,))
    de = de_X
    for nl in hidden_layers[::-1]:
        de = keras.layers.Dense(nl, activation='relu')(de)
    de_final = keras.layers.Dense(input_dims, activation='sigmoid')(de)
    decoder = keras.Model(de_X, de_final)

    # The AutoEncoder
    enc = encoder(X)
    dec = decoder(enc)
    auto = keras.Model(X, dec)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
