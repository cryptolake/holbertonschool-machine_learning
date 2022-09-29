#!/usr/bin/env python3
"""Resnet50."""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Resnet50."""
    init = K.initializers.HeNormal()
    input_layer = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(64, 7, strides=2,
                            padding='same',
                            kernel_initializer=init)(input_layer)
    batch_norm = K.layers.BatchNormalization()(conv1)

    activation = K.layers.Activation('relu')(batch_norm)

    max_pooling = K.layers.MaxPooling2D(pool_size=(3, 3), strides=2,
                                        padding='same')(activation)

    # ------------- convolution --------------------#
    CP1 = projection_block(max_pooling, [64, 64, 256], s=1)
    CI2 = identity_block(CP1, [64, 64, 256])
    CI3 = identity_block(CI2, [64, 64, 256])
    CP2 = projection_block(CI3, [128, 128, 512])

    CI4 = identity_block(CP2, [128, 128, 512])
    CI5 = identity_block(CI4, [128, 128, 512])
    CI6 = identity_block(CI5, [128, 128, 512])

    CP3 = projection_block(CI6, [256, 256, 1024])
    CI7 = identity_block(CP3, [256, 256, 1024])
    CI8 = identity_block(CI7, [256, 256, 1024])
    CI9 = identity_block(CI8, [256, 256, 1024])
    CI10 = identity_block(CI9, [256, 256, 1024])
    CI11 = identity_block(CI10, [256, 256, 1024])

    CP4 = projection_block(CI11, [512, 512, 2048])
    CI12 = identity_block(CP4, [512, 512, 2048])
    CI13 = identity_block(CI12, [512, 512, 2048])

    avg = K.layers.AveragePooling2D(pool_size=(7, 7)
                                    )(CI13)

    output_layer = K.layers.Dense(1000,
                                  kernel_initializer=init,
                                  activation='softmax')(avg)

    model = K.Model(input_layer, output_layer)
    return model
