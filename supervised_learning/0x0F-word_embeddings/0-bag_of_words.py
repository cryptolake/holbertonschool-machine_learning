#!/usr/bin/env python3
"""Bag of words embedding."""
import numpy as np
import re


def clean_strings(string):
    """Clean strings for processing."""
    lower_str = string.lower()
    symb_split = re.sub('[.!?",:;]', "", lower_str).split(' ')
    final_str = ' '.join(list(map(lambda string: re.sub("'.*$", "", string),
                                  symb_split)))
    return final_str


def bag_of_words(sentences, vocab=None):
    """
    Perform Bag of words embedding.

    sentences: the sentences to perform bow on
    vocab: the vocabulary to use if none use all words in sentences
    """
    if vocab is None:
        vocab = ' '.join(sentences)
        vocab = np.unique(clean_strings(vocab).split(' '))
        print(vocab)
    embeddings = np.zeros(shape=(len(sentences), len(vocab)))
    for i, sentence in enumerate(sentences):
        words = clean_strings(sentence).split(' ')
        for v, word in enumerate(vocab):
            embeddings[i, v] = words.count(word)
    return embeddings, vocab
