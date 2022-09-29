#!/usr/bin/env python3
"""Identity block for resnet."""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """Identity block for resnet."""
    init = K.initializers.he_normal()
    C1 = K.layers.Conv2D(filters[0], 1, 1, padding='same',
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
    AD = K.layers.Add()([A_prev, BN3])
    FAC = K.layers.Activation('relu')(AD)
    return FAC

