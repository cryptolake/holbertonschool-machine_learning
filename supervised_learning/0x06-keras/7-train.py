#!/usr/bin/env python3
"""Train keras Model."""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, verbose=True,
                shuffle=False):
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

    early_stopping is a boolean that indicates whether
    early stopping should be used

    patience is the patience used for early stopping

    learning_rate_decay is a boolean that indicates whether learning rate
    decay should be used

    alpha is the initial learning rate

    decay_rate is the decay rate

    shuffle is a boolean that determines whether to
    shuffle the batches every epoch.

    Normally, it is a good idea to shuffle, but for reproducibility,
    we have chosen to set the default to False.

    Returns: the History object generated after training the model
    """

    def schedule(epoch):
        previous_lr = 1

        def lr(epoch, start_lr, decay):
            nonlocal previous_lr
            previous_lr *= (start_lr / (1. + decay * epoch))
            return previous_lr
        return lr(epoch, alpha, decay_rate)

    callbacks = []
    if validation_data:
        if early_stopping:
            early_stopping_callback = K.callbacks.EarlyStopping(
                'val_loss', patience=patience)
            callbacks.append(early_stopping_callback)
        if learning_rate_decay:
            lr_callback = K.callbacks.LearningRateScheduler(
                schedule, verbose=True)
            callbacks.append(lr_callback)

    history = network.fit(data, labels, batch_size, epochs,
                          verbose=verbose, shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=callbacks)
    return history
