#!/usr/bin/env python3
"""Full training loop."""

import tensorflow.compat.v1 as tf


calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Training loop:

    X_train: a numpy.ndarray containing the training input data
    Y_train: a numpy.ndarray containing the training labels
    X_valid: a numpy.ndarray containing the validation input data
    Y_valid: a numpy.ndarray containing the validation labels
    layer_sizes: a list containing the number of nodes in
    each layer of the network
    activations: a list containing the activation functions
    for each layer of the network
    alpha: the learning rate
    iterations: the number of iterations to train over
    save_path: where to save the model
    """

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    accuracy = calculate_accuracy(y, y_pred)

    loss = calculate_loss(y, y_pred)

    train_op = create_train_op(loss, alpha)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init, feed_dict={x: X_train, y: Y_train})
    t_loss, t_accur = sess.run((loss, accuracy), feed_dict={
                               x: X_train, y: Y_train})
    v_loss, v_accur = sess.run((loss, accuracy), feed_dict={
        x: X_valid, y: Y_valid})

    print(f"""After 0 iterations:
\tTraining Cost: {t_loss}
\tTraining Accuracy: {t_accur}
\tValidation Cost: {v_loss}
\tValidation Accuracy: {v_accur}""")
    for i in range(iterations):
        if (i+1) % 100 == 0:
            print(f"""After {i+1} iterations:
\tTraining Cost: {t_loss}
\tTraining Accuracy: {t_accur}
\tValidation Cost: {v_loss}
\tValidation Accuracy: {v_accur}""")
        _, t_loss, t_accur = sess.run((train_op, loss, accuracy), feed_dict={
                                      x: X_train, y: Y_train})
        v_loss, v_accur = sess.run((loss, accuracy), feed_dict={
            x: X_valid, y: Y_valid})
    return saver.save(sess, save_path, iterations)
