#!/usr/bin/env python3
"""
Build a Convolutional Autoencoder, It's the same as the vanilla one, but
the hidden units are convolutions which makes this type of autoencoders
suitable for images.
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Build a convolutional autoencoder.
    Args:
        input_dims: input dimension which is the same as the output
        filters: list of number of filters in each hidden conv layer
        latent_dims: dimension of the latent space layer
    Return:
        Model: the autoencoder keras model
    """

    # The encoder
    X = keras.Input((*input_dims,))
    en = X
    for nf in filters:
        en = keras.layers.Conv2D(nf, kernel_size=(3, 3),
                                 activation='relu', padding='same')(en)
        en = keras.layers.MaxPool2D((2, 2), padding='same')(en)
    encoder = keras.Model(X, en)

    # The Decoder
    de_X = keras.Input((*latent_dims,))
    de = de_X
    for nf in reversed(filters[1:]):
        de = keras.layers.Conv2D(nf, kernel_size=(3, 3), padding='same',
                                 activation='relu')(de)
        de = keras.layers.UpSampling2D((2, 2))(de)
    de = keras.layers.Conv2D(filters[0], kernel_size=(3, 3), padding='valid',
                             activation='relu')(de)
    de = keras.layers.UpSampling2D((2, 2))(de)
    de_final = keras.layers.Conv2D(input_dims[-1], kernel_size=(3, 3),
                                   padding='same',
                                   activation='sigmoid')(de)
    decoder = keras.Model(de_X, de_final)

    # The AutoEncoder
    enc = encoder(X)
    dec = decoder(enc)
    auto = keras.Model(X, dec)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
