#!/usr/bin/env python3
"""Transition layer in DenseNet."""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Transition layer in DenseNet."""
    init = K.initializers.HeNormal()
    BN = K.layers.BatchNormalization()(X)
    AC = K.layers.Activation('relu')(BN)
    CONV = K.layers.Conv2D(int(nb_filters*compression), 1, 1, padding='same',
                           kernel_initializer=init)(AC)
    AP = K.layers.AveragePooling2D(2, int(compression**-1))(CONV)
    return AP, int(nb_filters*compression)
