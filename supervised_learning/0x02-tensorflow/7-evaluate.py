#!/usr/bin/env python3
"""Evaluate tensorflow model"""

import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """Evaluate tensorflow model"""
    tf.reset_default_graph()
    y_pred = tf.get_variable('y_pred', shape=(None, 10))
    accuracy = tf.get_variable('accuracy', shape=())
    loss = tf.get_variable('loss', shape=())
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, save_path)
        return y_pred.eval(), accuracy.eval(), loss.eval()
