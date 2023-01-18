#!/usr/bin/env python3
"""Cumulative N-gram BLEU score."""
import numpy as np


def transform_multigram(sentence, n):
    """Transform list of words into list of tuples of words."""
    new_sentence = []
    for i in range(len(sentence)-n+1):
        grams = []
        for j in range(i, i+n):
            grams.append(sentence[j])
        new_sentence.append(tuple(grams))

    return new_sentence


def clipped_precision(references, sentence, n):
    """N-gram BLEU score."""
    n_references = list(map(lambda x: transform_multigram(x, n), references))
    n_sentence = transform_multigram(sentence, n)
    n_grams = list(map(lambda x: tuple(x), np.unique(n_sentence, axis=0)))
    P = 0

    for gram in n_grams:
        count = max(map(lambda x: x.count(gram), n_references))
        P += count
    P /= len(n_sentence)

    return P


def cumulative_bleu(references, sentence, n):
    """Cumulative N-gram BLEU SCORE."""
    GAP = 1
    for i in range(1, n+1):
        GAP *= clipped_precision(references, sentence, i)**(1/n)

    c = len(sentence)
    r = min(map(lambda x: len(x), references))
    brevity = 1
    if c <= r:
        brevity = np.exp(1-(r/c))
    BLEU = brevity * GAP
    return BLEU
