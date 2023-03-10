#!/usr/bin/env python3
"""
Build a Variational Autoencoder, in a variational Autoencoder we wish to
generate new data points from the vector space of our original data,
we want the generated data to be similar but not the same.
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Build a Variational Autoencoder.

    Args:
        input_dims : input dimension which is the same as output
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
    en_final = keras.layers.Dense(latent_dims, activation='relu')(en)
    variance = keras.layers.Dense(latent_dims, activation=None)(en_final)
    mean = keras.layers.Dense(latent_dims, activation=None)(en_final)
    encoder = keras.Model(X, [en_final, mean, variance])

    # The Decoder
    de_X = keras.Input((latent_dims,))
    de = de_X
    for nl in hidden_layers[::-1]:
        de = keras.layers.Dense(nl, activation='relu')(de)
    de_final = keras.layers.Dense(input_dims, activation='sigmoid')(de)
    decoder = keras.Model(de_X, de_final)

    def sampler(params):
        mu, logstd = params
        rand = keras.backend.random_normal((keras.backend.shape(mu)[0],
                                           latent_dims))
        return mu + keras.backend.exp(logstd) * rand

    # The AutoEncoder
    enc, mu, sig = encoder(X)
    enc_samples = keras.layers.Lambda(sampler)((mu, sig))
    dec = decoder(enc_samples)
    auto = keras.Model(X, dec)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
