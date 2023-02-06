#!/usr/bin/env python3
"""Load and prepare dataset for nlp task."""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Load and prepare dataset for nlp task."""

    def __init__(self):
        """Initialize Dataset."""
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train',
            as_supervised=True
        )
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True
        )
        (self.tokenizer_pt,
            self.tokenizer_en) = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """Tokenize dataset."""
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder\
            .build_from_corpus(
                (pt.numpy() for pt, _ in data),
                target_vocab_size=2**15)

        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder\
            .build_from_corpus(
                (en.numpy() for _, en in data),
                target_vocab_size=2**15)

        return tokenizer_pt, tokenizer_en
