#!/usr/bin/env python3
"""Evaluate tensorflow model."""

import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """Evaluate tensorflow model."""
    saver = tf.train.import_meta_graph(save_path+'.meta')
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        y_pred = tf.get_collection('y_pred')[0]
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        pred, t_loss, t_accur = sess.run((y_pred, loss, accuracy),
                                         feed_dict={x: X, y: Y})
        return pred, t_accur, t_loss
