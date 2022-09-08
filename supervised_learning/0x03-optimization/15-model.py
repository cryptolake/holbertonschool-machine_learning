#!/usr/bin/env python3
"""Build model with all optimizations."""
import tensorflow.compat.v1 as tf
import numpy as np


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Implement Adam with tensorflow.

    loss is the loss of the network
    alpha is the learning rate
    beta1 is the weight used for the first moment
    beta2 is the weight used for the second moment
    epsilon is a small number to avoid division by zero
    Returns: the Adam optimization operatio
    """
    train = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    grads = train.compute_gradients(loss)
    print('\n\n\n\n\n\n\n\n', grads, '\n\n\n\n\n')
    return train.apply_gradients(grads)


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.

    X is the first numpy.ndarray of shape (m, nx) to shuffle
        m is the number of data points
        nx is the number of features in X
    Y is the second numpy.ndarray of shape (m, ny) to shuffle
        m is the same number of data points as in X
        ny is the number of features in Y
    Returns: the shuffled X and Y matrices
    """
    ind = np.random.permutation(len(X))
    return X[ind], Y[ind]


def calculate_accuracy(y, y_pred):
    """Accuracy of prediction."""
    y = tf.argmax(y, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)
    correct = tf.equal(y, y_pred)
    return tf.reduce_mean(tf.cast(correct, dtype='float'))


def create_layer(prev, n, activation):
    """Create tensorflow layer."""
    weight = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(n, activation=activation, name="layer",
                                  kernel_initializer=weight)
    return layer(prev)


def create_batch_norm_layer(prev, n, activation):
    """
    Batch Normalization Layer in tensorflow.

    prev is the activated output of the previous layer
    n is the number of nodes in the layer to be created
    activation is the activation function that should
    be used on the output of the layer
    Returns: a tensor of the activated output for the layer
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(n, kernel_initializer=init)
    z = layer(prev)
    gamma = tf.Variable(1., trainable=True)
    beta = tf.Variable(0., trainable=True)
    mean = tf.math.reduce_mean(z, axis=0)
    var = tf.math.reduce_variance(z, axis=0)
    epsilon = 1e-8
    normalized = tf.nn.batch_normalization(z, mean, var, beta, gamma, epsilon)
    return activation(normalized)


def forward_prop(x, layer_sizes=[], activations=[]):
    """Forward propagation."""
    xi = None
    for layer, activation in zip(layer_sizes, activations):
        if activation is None:
            xi = create_layer(x, layer, activation)
        else:
            xi = create_batch_norm_layer(x, layer, activation)
        x = xi

    return xi


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """
    Build, trains, and saves a neural network model in tensorflow using,\
    Adam optimization, mini-batch gradient descent, learning rate decay,\
    and batch normalization.

    Data_train is a tuple containing
    the training inputs and training labels, respectively

    Data_valid is a tuple containing the validation
    inputs and validation labels, respectively

    layers is a list containing the number of nodes
    in each layer of the network

    activation is a list containing the activation functions
    used for each layer of the network

    alpha is the learning rate

    beta1 is the weight for the first moment of Adam Optimization

    beta2 is the weight for the second moment of Adam Optimization

    epsilon is a small number used to avoid division by zero

    decay_rate is the decay rate for inverse time decay of the learning rate
    (the corresponding decay step should be 1)

    batch_size is the number of data points that should be in a mini-batch

    epochs is the number of times the training should pass
    through the whole dataset

    save_path is the path where the model should be saved to

    Returns: the path where the model was saved
    """
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    x = tf.placeholder(tf.float32, shape=(None, X_train.shape[1]))
    tf.add_to_collection('x', x)

    y = tf.placeholder(tf.float32, shape=(None, Y_train.shape[1]))
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    tf.add_to_collection('loss', loss)

    global_step = tf.Variable(0, trainable=False)
    decay_step = tf.Variable(X_train.shape[0]//batch_size)

    alpha = tf.train.inverse_time_decay(alpha, global_step,
                                        decay_step, decay_rate,
                                        staircase=True)

    train = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    train_op = train.minimize(loss, global_step)

    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            i = 1
            t_loss, t_accur = sess.run((loss, accuracy),
                                       feed_dict={x: X_train, y: Y_train})
            v_loss, v_accur = sess.run((loss, accuracy),
                                       feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(t_loss))
            print("\tTraining Accuracy: {}".format(t_accur))
            print("\tValidation Cost: {}".format(v_loss))
            print("\tValidation Accuracy: {}".format(v_accur))
            X_shuffle, Y_shuffle = shuffle_data(X_train, Y_train)
            for b in range(0, len(X_train), batch_size):
                X_batch = X_shuffle[b:b+batch_size, ]
                Y_batch = Y_shuffle[b:b+batch_size, ]
                _ = sess.run((train_op), feed_dict={x: X_batch, y: Y_batch,
                                                    global_step: epoch})
                if i % 100 == 0:
                    t_loss, t_accur = sess.run((loss, accuracy),
                                               feed_dict={x: X_batch,
                                                          y: Y_batch})
                    print("\tStep {}:".format(i))
                    print("\t\tCost: {}".format(t_loss))
                    print("\t\tAccuracy: {}".format(t_accur))
                i += 1
        t_loss, t_accur = sess.run((loss, accuracy),
                                   feed_dict={x: X_train, y: Y_train})
        v_loss, v_accur = sess.run((loss, accuracy),
                                   feed_dict={x: X_valid, y: Y_valid})
        print("After {} epochs:".format(epoch+1))
        print("\tTraining Cost: {}".format(t_loss))
        print("\tTraining Accuracy: {}".format(t_accur))
        print("\tValidation Cost: {}".format(v_loss))
        print("\tValidation Accuracy: {}".format(v_accur))

        return saver.save(sess, save_path)
