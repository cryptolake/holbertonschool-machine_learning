#!/usr/bin/env python3
"""Mini Batch optimization."""

import tensorflow.compat.v1 as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Train a loaded neural network model using mini-batch gradient descent.

    X_train is a numpy.ndarray of shape (m, 784) containing the training data

        m is the number of data points
        784 is the number of input features

    Y_train is a one-hot numpy.ndarray of shape (m, 10)
    containing the training labels

        10 is the number of classes the model should classify

    X_valid is a numpy.ndarray of shape (m, 784)
    containing the validation data

    Y_valid is a one-hot numpy.ndarray of shape (m, 10)
    containing the validation labels

    batch_size is the number of data points in a batch

    epochs is the number of times the training should pass
    through the whole dataset

    load_path is the path from which to load the model

    save_path is the path to where the model should be saved after training

    Returns: the path where the model was saved
    """
    with tf.Session() as sess:
        graph = tf.train.import_meta_graph(load_path+'.meta')
        saver = tf.train.Saver()
        graph.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        for epoch in range(epochs):
            i = 1
            t_loss, t_accur = sess.run((loss, accuracy),
                                       feed_dict={x: X_train,
                                                  y: Y_train})
            v_loss, v_accur = sess.run((loss, accuracy),
                                       feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(t_loss))
            print("\tTraining Accuracy: {}".format(t_accur))
            print("\tValidation Cost: {}".format(v_loss))
            print("\tValidation Accuracy: {}".format(v_accur))
            X_train, Y_train = shuffle_data(X_train, Y_train)
            for b in range(0, len(X_train), batch_size):
                X_batch = X_train[b:b+batch_size, ]
                Y_batch = Y_train[b:b+batch_size, ]
                _ = sess.run(train_op, feed_dict={x: X_batch,
                                                  y: Y_batch})
                if i % 100 == 0:
                    t_loss, t_accur = sess.run((loss, accuracy),
                                               feed_dict={x: X_batch,
                                                          y: Y_batch})
                    print("\tStep {}:".format(i))
                    print("\t\tCost: {}".format(t_loss))
                    print("\t\tAccuracy: {}".format(t_accur))
                i += 1
        t_loss, t_accur = sess.run((loss, accuracy),
                                   feed_dict={x: X_train,
                                              y: Y_train})
        v_loss, v_accur = sess.run((loss, accuracy),
                                   feed_dict={x: X_valid, y: Y_valid})
        print("After {} epochs:".format(epoch+1))
        print("\tTraining Cost: {}".format(t_loss))
        print("\tTraining Accuracy: {}".format(t_accur))
        print("\tValidation Cost: {}".format(v_loss))
        print("\tValidation Accuracy: {}".format(v_accur))
        return saver.save(sess, save_path)
