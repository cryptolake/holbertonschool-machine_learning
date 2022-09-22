#!/usr/bin/env python3
"""LeNet-5 in Keras."""
import tensorflow.keras as K


def lenet5(X):
    """Create LeNet-5 in Keras."""
    init = K.initializers.VarianceScaling(scale=2.0)
    _, x, y, k = X.shape
    model = K.Sequential([
        K.layers.Conv2D(6, 5, kernel_initializer=init, padding='same',
                        activation='relu', input_shape=(x, y, k)),
        K.layers.MaxPool2D((2, 2), (2, 2)),
        K.layers.Conv2D(16, 5, kernel_initializer=init, padding='valid',
                        activation='relu'),
        K.layers.MaxPool2D((2, 2), (2, 2)),
        K.layers.Flatten(),
        K.layers.Dense(120, activation='relu', kernel_initializer=init),
        K.layers.Dense(84, activation='relu', kernel_initializer=init),
        K.layers.Dense(10, activation='softmax', kernel_initializer=init)
    ])
    adam = K.optimizers.Adam()
    loss = K.losses.CategoricalCrossentropy()
    model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
    return model
