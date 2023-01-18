#!/usr/bin/env python3
"""Word2Vec python."""
from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5,
                   negative=5, cbow=True, iterations=5, seed=0, workers=1):
    """Initialize ant Train a word2vec model."""
    sg = 1
    if cbow:
        sg = 0
    model = Word2Vec(size=size,
                     window=window, min_count=min_count, workers=workers,
                     negative=negative, seed=seed, sg=sg)
    model.build_vocab(sentences)
    model.train(sentences, epochs=iterations,
                total_examples=model.corpus_count)
    return model
