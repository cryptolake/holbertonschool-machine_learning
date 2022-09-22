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


def create_pooling_layer(prev, size, stride):
    """Create pooling layer."""
    layer = tf.layers.MaxPooling2D(size, stride)
    return layer(prev)


def create_conv_layer(prev, size, padding, stride):
    """Create Conv layer."""
    weight = tf.keras.initializers.VarianceScaling(scale=2.0)
    layer = tf.layers.Conv2D(filters=size[2],
                             kernel_size=(size[0], size[1]), strides=stride,
                             padding=padding, kernel_initializer=weight,
                             activation='relu')
    return layer(prev)


def create_normal_layer(prev, n, activation):
    """Create tensorflow layer."""
    weight = tf.keras.initializers.VarianceScaling(scale=2.0)
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=weight)
    return layer(prev)


def conv_forward_prop(x, conv_sizes=[], padding=[], pool_sizes=[], strides=[]):
    """Convolution forward propagation."""
    xi = None
    for conv, pad, pool, stride in zip(conv_sizes, padding,
                                       pool_sizes, strides):
        xi = create_conv_layer(x, conv, pad, stride[0])
        xi = create_pooling_layer(xi, pool, stride[1])
        x = xi
    return xi


def forward_prop(x, layer_sizes=[], activations=[]):
    """Forward propagation."""
    xi = None
    for layer, activation in zip(layer_sizes, activations):
        xi = create_normal_layer(x, layer, activation)
        x = xi
    return xi


def calculate_loss(y, y_pred):
    """Calculate cross entropy loss."""
    return tf.losses.softmax_cross_entropy(y, y_pred)


def calculate_accuracy(y, y_pred):
    """Accuracy of prediction."""
    y = tf.argmax(y, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)
    correct = tf.equal(y, y_pred)
    return tf.reduce_mean(tf.cast(correct, dtype='float'))


def create_train_op(loss):
    """Create training operation."""
    train = tf.train.AdamOptimizer()
    grads = train.compute_gradients(loss)
    return train.apply_gradients(grads)


def lenet5(x, y):
    """
    Build a modified LeNet-5 model for number recognition.

    x is a tf.placeholder of shape (m, 28, 28, 1)
    containing the input images for the network
        m is the number of images
    y is a tf.placeholder of shape (m, 10)
    containing the one-hot labels for the network
    """
    conv_sizes = [(5, 5, 6), (5, 5, 16)]
    pool_sizes = [(2, 2), (2, 2)]
    strides = [((1, 1), (2, 2)), ((1, 1), (2, 2))]
    paddings = ['same', 'valid']
    layer_sizes = [120, 84, 10]
    activations = ['relu', 'relu', None]
    y_conv = conv_forward_prop(x, conv_sizes, paddings, pool_sizes, strides)
    x_flat = tf.layers.Flatten()(y_conv)
    y_pred = forward_prop(x_flat, layer_sizes, activations)
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)
    train_op = create_train_op(loss)

    return y_pred, train_op, loss, accuracy
