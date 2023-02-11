#!/usr/bin/env python3
"""Load and prepare dataset for nlp task."""
import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset:
    """Load and prepare dataset for nlp task."""

    def __init__(self, batch_size, max_len):
        """Initialize Dataset."""
        (data_train, data_valid), ds_info = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split=['train', 'validation'],
            as_supervised=True,
            with_info=True
        )

        def filter_ds(x, y):
            """Filter dataset."""
            return tf.math.logical_and(tf.size(x) <= max_len,
                                       tf.size(y) <= max_len)

        (self.tokenizer_pt,
            self.tokenizer_en) = self.tokenize_dataset(data_train)

        self.data_train_size = ds_info.splits['train'].num_examples
        self.data_valid_size = ds_info.splits['validation'].num_examples

        padding = ([max_len], [max_len])
        data_train = data_train.map(self.tf_encode)
        data_train = data_train.filter(filter_ds)
        data_train = data_train.cache()
        data_train = data_train.shuffle(self.data_train_size)
        data_train = data_train.padded_batch(batch_size, padded_shapes=padding)
        self.data_train = data_train.prefetch(tf.data.AUTOTUNE)

        data_valid = data_valid.map(self.tf_encode)
        data_valid = data_valid.filter(filter_ds)
        data_valid = data_valid.shuffle(self.data_valid_size)
        self.data_valid = data_valid.padded_batch(batch_size,
                                                  padded_shapes=padding)

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
        pt_vocab_size = self.tokenizer_pt.vocab_size
        en_vocab_size = self.tokenizer_en.vocab_size

        pt_tokens = [pt_vocab_size] +\
            self.tokenizer_pt.encode(pt.numpy()) + [pt_vocab_size+1]
        en_tokens = [en_vocab_size] +\
            self.tokenizer_en.encode(en.numpy()) + [en_vocab_size+1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """Wrap encode function."""
        return tf.py_function(func=self.encode, inp=[pt, en],
                              Tout=(tf.int64, tf.int64))
