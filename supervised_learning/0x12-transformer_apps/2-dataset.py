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

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

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

    def encode(self, pt, en):
        """Encode sentences."""
        def add_tokens(tokens, size):
            """Add the START and END tokens."""
            tokens.insert(0, size)
            tokens.append(size+1)
            return tokens

        pt_vocab_size = self.tokenizer_pt.vocab_size
        en_vocab_size = self.tokenizer_en.vocab_size

        pt_tokens = self.tokenizer_pt.encode(pt.numpy())
        en_tokens = self.tokenizer_en.encode(en.numpy())

        return (add_tokens(pt_tokens, pt_vocab_size),
                add_tokens(en_tokens, en_vocab_size))

    def tf_encode(self, pt, en):
        """Wrap encode function."""
        return tf.py_function(func=self.encode, inp=[pt, en],
                              Tout=(tf.int64, tf.int64))
