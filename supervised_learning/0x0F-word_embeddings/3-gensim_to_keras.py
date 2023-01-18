#!/usr/bin/env python3
"""Word2Vec gensim to Keras layer."""
from tensorflow.keras.layers import Embedding


def gensim_to_keras(model):
    """Word2Vec gensim to Keras layer."""
    keyed_vectors = model.wv
    weights = keyed_vectors.vectors

    layer = Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )
    return layer
