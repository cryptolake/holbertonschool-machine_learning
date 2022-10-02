#!/usr/bin/env python3
"""Dense Block."""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Dense Block."""
    init = K.initializers.HeNormal()
    prevs = X
    for _ in range(layers):
        BN = K.layers.BatchNormalization()(prevs)
        AC = K.layers.Activation('relu')(BN)
        L = K.layers.Conv2D(4*growth_rate, 1, 1, padding='same',
                            kernel_initializer=init)(AC)
        BN = K.layers.BatchNormalization()(L)
        AC = K.layers.Activation('relu')(BN)
        L = K.layers.Conv2D(growth_rate, 3, 1, padding='same',
                            kernel_initializer=init)(AC)
        prevs = K.layers.Concatenate(axis=3)([prevs, L])
        nb_filters += growth_rate
    return prevs, nb_filters
