#!/usr/bin/env python3
"""Unigram BLEU score."""
import numpy as np


def uni_bleu(refrences, sentence):
    """Unigram BLEU score."""
    uni_grams = np.unique(sentence)
    P = 0

    for gram in uni_grams:
        count = max(map(lambda x: x.count(gram), refrences))
        P += count
    c = len(sentence)
    r = min(map(lambda x: len(x), refrences))
    P /= len(sentence)
    brevity = 1
    if c <= r:
        brevity = np.exp(1-(r/c))
    BLEU = brevity * P

    return BLEU
