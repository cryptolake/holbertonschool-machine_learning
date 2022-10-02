#!/usr/bin/env python3
"""Full Dense net."""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Densenet 121 with keras."""
    init = K.initializers.HeNormal()
    INPUT = K.Input((224, 224, 3))
    BN1 = K.layers.BatchNormalization()(INPUT)
    AC1 = K.layers.Activation('relu')(BN1)
    CONV1 = K.layers.Conv2D(64, 7, 2, padding='same',
                            kernel_initializer=init)(AC1)
    MP1 = K.layers.MaxPool2D(3, 2, padding='same')(CONV1)
    DB1, DBS1 = dense_block(MP1, 64, growth_rate, 6)
    TL1, TLS1 = transition_layer(DB1, DBS1, compression)
    DB2, DBS2 = dense_block(TL1, TLS1, growth_rate, 12)
    TL2, TLS2 = transition_layer(DB2, DBS2, compression)
    DB3, DBS3 = dense_block(TL2, TLS2, growth_rate, 24)
    TL3, TLS3 = transition_layer(DB3, DBS3, compression)
    DB4, _ = dense_block(TL3, TLS3, growth_rate, 16)
    GAP = K.layers.AveragePooling2D(7)(DB4)
    SOFT = K.layers.Dense(1000, activation='softmax',
                          kernel_initializer=init)(GAP)
    return K.Model(inputs=INPUT, outputs=SOFT)
