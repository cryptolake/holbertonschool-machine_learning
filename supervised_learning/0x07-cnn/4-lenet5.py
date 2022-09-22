#!/usr/bin/env python3
"""
Create The LeNet-5 Convlutional neural network.

The model should consist of the following layers in order:
    Convolutional layer with 6 kernels of shape 5x5 with same padding
    Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Convolutional layer with 16 kernels of shape 5x5 with valid padding
    Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Fully connected layer with 120 nodes
    Fully connected layer with 84 nodes
    Fully connected softmax output layer with 10 nodes
"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """Accuracy of prediction."""
    y = tf.argmax(y, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)
    correct = tf.equal(y, y_pred)
    return tf.reduce_mean(tf.cast(correct, dtype='float'))


def lenet5(x, y):
    """
    Build a modified LeNet-5 model for number recognition.

    x is a tf.placeholder of shape (m, 28, 28, 1)
    containing the input images for the network
        m is the number of images
    y is a tf.placeholder of shape (m, 10)
    containing the one-hot labels for the network
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0)
    x1 = tf.layers.Conv2D(filters=6, kernel_size=5, padding='same',
                          kernel_initializer=init, activation='relu')(x)
    px1 = tf.layers.MaxPooling2D(2, 2)(x1)
    x2 = tf.layers.Conv2D(filters=16, kernel_size=5, padding='valid',
                          kernel_initializer=init, activation='relu')(px1)
    px2 = tf.layers.MaxPooling2D(2, 2)(x2)
    x_flat = tf.layers.Flatten()(px2)
    cx1 = tf.layers.Dense(120, activation='relu',
                          kernel_initializer=init)(x_flat)
    cx2 = tf.layers.Dense(84, activation='relu', kernel_initializer=init)(cx1)
    cx3 = tf.layers.Dense(10, kernel_initializer=init)(cx2)
    y_pred = tf.nn.softmax(cx3)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=cx3)
    train_op = tf.train.AdamOptimizer().minimize(loss)
    accuracy = calculate_accuracy(y, y_pred)

    return y_pred, train_op, loss, accuracy
