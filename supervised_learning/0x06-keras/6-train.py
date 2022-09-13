#!/usr/bin/env python3
"""Train keras Model."""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, 
                validation_data=None, early_stopping=False, 
                patience=0, verbose=True, shuffle=False):
    """
    Train Keras Model.

    network is the model to train

    data is a numpy.ndarray of shape (m, nx) containing the input data

    labels is a one-hot numpy.ndarray of shape (m, classes)
     containing the labels of data

    batch_size is the size of the batch used for mini-batch gradient descent

    epochs is the number of passes through data for mini-batch gradient descent

    verbose is a boolean that determines if output should be
    printed during training

    validation_data is the data to validate the model with, if not None

    early_stopping is a boolean that indicates whether early stopping should be used

    patience is the patience used for early stopping

    shuffle is a boolean that determines whether to
    shuffle the batches every epoch.

    Normally, it is a good idea to shuffle, but for reproducibility,
    we have chosen to set the default to False.

    Returns: the History object generated after training the model
    """
    if validation_data:
        early_stopping_callback = K.callbacks.EarlyStopping(
            'val_loss', patience=patience)
    else:
        early_stopping_callback = None
    history = network.fit(data, labels, batch_size, epochs,
                          verbose=verbose, shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=early_stopping_callback)
    return history
