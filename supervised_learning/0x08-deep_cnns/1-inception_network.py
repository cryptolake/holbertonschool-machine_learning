#!/usr/bin/env python3
"""Inception network."""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Inception network."""
    init = K.initializers.HeNormal()
    lin = K.Input((224, 224, 3))
    C1 = K.layers.Conv2D(64, 7, 2, activation='relu',
                         padding='same', kernel_initializer=init)(lin)
    m1 = K.layers.MaxPool2D((3, 3), 2, padding='same')(C1)
    C2 = K.layers.Conv2D(192, 3, 1, activation='relu',
                         padding='same', kernel_initializer=init)(m1)
    m2 = K.layers.MaxPool2D((3, 3), 2, padding='same')(C2)
    b1 = inception_block(m2, [64, 96, 128, 16, 32, 32])
    b2 = inception_block(b1, [128, 128, 192, 32, 96, 64])
    m3 = K.layers.MaxPool2D((3, 3), 2, padding='same')(b2)
    b3 = inception_block(m3, [192, 96, 208, 16, 48, 64])
    b4 = inception_block(b3, [160, 112, 224, 24, 64, 64])
    b5 = inception_block(b4, [128, 128, 256, 24, 64, 64])
    b6 = inception_block(b5, [112, 144, 288, 32, 64, 64])
    b7 = inception_block(b6, [256, 160, 320, 32, 128, 128])
    m4 = K.layers.MaxPooling2D((3, 3), 2, padding='same')(b7)
    b8 = inception_block(m4, [256, 160, 320, 32, 128, 128])
    b9 = inception_block(b8, [384, 192, 384, 48, 128, 128])
    avg1 = K.layers.AveragePooling2D((7, 7), 1)(b9)
    d1 = K.layers.Dropout(0.4)(avg1)
    softmax = K.layers.Dense(1000, activation='softmax')(d1)
    model = K.models.Model(lin, softmax)
    return model
