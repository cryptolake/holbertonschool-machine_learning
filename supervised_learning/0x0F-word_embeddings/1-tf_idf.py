#!/usr/bin/env python3
"""Bag of words TF-IDF embedding."""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Perform Bag of words TF-IDF embedding.

    sentences: the sentences to perform bow on
    vocab: the vocabulary to use if none use all words in sentences
    """
    cont_vector = TfidfVectorizer(vocabulary=vocab)
    embeddings = cont_vector.fit_transform(sentences)
    features = cont_vector.get_feature_names()
    return embeddings.toarray(), features
