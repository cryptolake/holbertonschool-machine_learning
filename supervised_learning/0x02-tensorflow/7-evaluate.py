#!/usr/bin/env python3
"""Evaluate tensorflow model."""

import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """Evaluate tensorflow model."""
    saver = tf.train.import_meta_graph(save_path+'.meta')
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        y_pred = tf.get_collection('y_pred')
        accuracy = tf.get_collection('accuracy')
        cost = tf.get_collection('loss')
        return y_pred[0], accuracy[0], cost[0]
