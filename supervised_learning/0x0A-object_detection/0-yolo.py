#!/usr/bin/env python3
"""Implementing YOLO."""
import tensorflow as tf


class Yolo:
    """
    Implementing Yolo class.

    Implement YOLO model, it's an object detection model,
    that is fully convoluted.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialize the class.

        model_path is the path to where a Darknet Keras model is stored

        classes_path is the path to where the list of class names used
        for the Darknet model, listed in order of index, can be found

        class_t is a float representing the box score threshold for
        the initial filtering step

        nms_t is a float representing the IOU threshold for non-max suppression

        anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
        """
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            classes = list(map(lambda x: x.replace('\n', ''), f.readlines()))
            self.class_names = classes
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
