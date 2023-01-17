#!/usr/bin/env python3
"""Bag of words embedding."""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    Perform Bag of words embedding.

    sentences: the sentences to perform bow on
    vocab: the vocabulary to use if none use all words in sentences
    """
    cont_vector = CountVectorizer(vocabulary=vocab)
    embeddings = cont_vector.fit_transform(sentences)
    features = cont_vector.get_feature_names()
    return embeddings.toarray(), features
