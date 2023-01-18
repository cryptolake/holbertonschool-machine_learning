#!/usr/bin/env python3
"""Fasttext model python."""
from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5,
                   negative=5, window=5, cbow=True, iterations=5,
                   seed=0, workers=1):
    """Initialize ant Train a word2vec model."""
    sg = int(not cbow)
    model = FastText(sentences=sentences, size=size,
                     window=window, min_count=min_count, workers=workers,
                     negative=negative, seed=seed, sg=sg)
    model.train(sentences, epochs=iterations,
                total_examples=model.corpus_count)
    return model
