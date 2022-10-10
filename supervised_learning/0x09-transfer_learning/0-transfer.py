#!/usr/bin/env python3
"""Implement my own transfer learning on cifar 10."""
import tensorflow.keras as K
import tensorflow as tf


def get_data():
    """Initialize cifar-10 data."""
    train, valid = K.datasets.cifar10.load_data()
    return train, valid


def preprocess_data(X, Y):
    """Preprocess Data."""
    Y = tf.one_hot(Y, 10)
    return (K.applications.resnet50.preprocess_input(X),
            tf.reshape(Y, [Y.shape[0], 10]))


def create_model():
    """Create model from resnet50."""
    base_model = K.applications.ResNet50(
            weights='imagenet',
            include_top=True,
            pooling='max')

    old_input = K.Input((32, 32, 3))
    pre = K.layers.Lambda(lambda x: tf.image.resize(x, [224, 224]))(old_input)
    inputs = base_model(pre)
    outputs = K.layers.Dense(10, activation='softmax')(inputs)
    model = K.Model(old_input, outputs)

    return model, base_model


if __name__ == "__main__":
    (X_train, Y_train), (X_valid, Y_valid) = get_data()

    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_valid, Y_valid = preprocess_data(X_valid, Y_valid)

    model, base_model = create_model()
    model.compile(optimizer=K.optimizers.Adam(1e-5),
                  loss=K.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.summary()
    model.fit(x=X_train, y=Y_train, validation_data=(X_valid, Y_valid),
              batch_size=64, epochs=2)
    base_model.trainable = False
    model.compile(optimizer=K.optimizers.Adam(),
                  loss=K.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.summary()
    model.fit(x=X_train, y=Y_train, validation_data=(X_valid, Y_valid),
              batch_size=64, epochs=4)
    model.save('cifar10.h5')
