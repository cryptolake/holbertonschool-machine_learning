#!/usr/bin/env python3
"""
Build a Sparse Autoencoder, It's the same as the vanilla one, but
instead of the latent space having less units we apply L1 regularization.
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Build an autoencoder.
    Args:
        input_dims: input dimension which is the same as the output
        hidden_layers: list number of nodes in each hidden layer
        latent_dims: dimension of the latent space layer
    Return:
        Model: the autoencoder keras model
    """

    # The encoder
    X = keras.Input((input_dims,))
    en = X
    for nl in hidden_layers:
        en = keras.layers.Dense(nl, activation='relu')(en)
    en_final = keras.layers.Dense(latent_dims, activation='relu',
                                  activity_regularizer=keras.regularizers.l1(
                                      lambtha))(en)
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
