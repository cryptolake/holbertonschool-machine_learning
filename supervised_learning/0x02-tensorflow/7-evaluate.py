#!/usr/bin/env python3
"""Evaluate tensorflow model"""

import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """Evaluate tensorflow model"""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path+'.meta')
        saver.restore(sess, save_path)
        y_pred = tf.get_collection('y_pred')
        accuracy = tf.get_collection('accuracy')
        loss = tf.get_collection('loss')
        return y_pred[0], accuracy[0], loss[0]
