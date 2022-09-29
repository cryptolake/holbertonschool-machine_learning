#!/usr/bin/env python3
"""Projection block for resnet."""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """Projection block for resnet."""
    init = K.initializers.HeNormal()
    C1 = K.layers.Conv2D(filters[0], 1, s, padding='same',
                         kernel_initializer=init)(A_prev)
    BN1 = K.layers.BatchNormalization()(C1)
    AC1 = K.layers.Activation('relu')(BN1)
    C2 = K.layers.Conv2D(filters[1], 3, 1, padding='same',
                         kernel_initializer=init)(AC1)
    BN2 = K.layers.BatchNormalization()(C2)
    AC2 = K.layers.Activation('relu')(BN2)
    C3 = K.layers.Conv2D(filters[2], 1, 1, padding='same',
                         kernel_initializer=init)(AC2)
    BN3 = K.layers.BatchNormalization()(C3)
    CPROJ = K.layers.Conv2D(filters[2], 1, s, padding='same',
                            kernel_initializer=init)(A_prev)
    BN4 = K.layers.BatchNormalization()(CPROJ)
    AD = K.layers.Add()([BN3, BN4])
    FAC = K.layers.Activation('relu')(AD)
    return FAC
