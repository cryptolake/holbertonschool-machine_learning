#!/usr/bin/env python3
"""Bag of words embedding."""
import numpy as np


def clean_string(string):
    """Clean strings for processing."""
    lower_str = string.lower().replace("'s", '')
    proc_sentence = ''.join([c for c in lower_str if c.isalpha() or c == ' '])
    return proc_sentence


def bag_of_words(sentences, vocab=None):
    """
    Perform Bag of words embedding.

    sentences: the sentences to perform bow on
    vocab: the vocabulary to use if none use all words in sentences
    """
    if vocab is None:
        vocab = ' '.join(sentences)
        vocab = list(np.unique(clean_string(vocab).split(' ')))
    embeddings = np.zeros(shape=(len(sentences), len(vocab)))
    for i, sentence in enumerate(sentences):
        words = clean_string(sentence).split(' ')
        for v, word in enumerate(vocab):
            embeddings[i, v] = words.count(word)
    return embeddings, vocab
