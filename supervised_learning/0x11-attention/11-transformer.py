#!/usr/bin/env python3
"""TRANSFORMER ASSEMBLE!!!!!!."""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """Transformer model implimentation in keras."""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """Initialize Model."""
        super().__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """Call Model."""
        # This is the context of the input sequence
        # (the relation between each word)
        # devived by self attention
        y_encoder = self.encoder(inputs, training, encoder_mask)
        y_decoder = self.decoder(target, y_encoder, training, look_ahead_mask,
                                 decoder_mask)

        Y = self.linear(y_decoder)

        return Y
