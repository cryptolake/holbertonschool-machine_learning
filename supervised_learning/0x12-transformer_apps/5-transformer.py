#!/usr/bin/env python3
"""Full transformer implimentation."""
import numpy as np
import tensorflow as tf


def positional_encoding(max_seq_len, dm):
    """Positional encoding."""
    P = np.zeros((max_seq_len, dm))
    n = 10000
    for k in range(max_seq_len):
        for i in range(dm//2):
            P[k, 2*i] = np.sin(k/(n**((2*i)/dm)))
            P[k, 2*i+1] = np.cos(k/(n**((2*i)/dm)))
    return P


def sdp_attention(Q, K, V, mask=None):
    """Attention as in the 'attention is all you need' paper."""
    qk_matmul = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(K.shape[-1], tf.float32)
    scaled = qk_matmul / tf.math.sqrt(dk)
    if mask is not None:
        scaled += (mask * -1e9)

    att_weights = tf.nn.softmax(scaled, axis=-1)
    sdp_att = tf.matmul(att_weights, V)

    return sdp_att, att_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi head attention as implemented in the paper."""

    def __init__(self, dm, h):
        """Initialize instance."""
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask):
        """Functionality of the layer."""
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        batch = tf.shape(Q)[0]

        # Split logic from this tutorial:
        # https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853
        def split(tensor):
            """
            Split tensor.

            from (batch, seq, emb) to (batch, head, seq, depth).
            """
            batch = tf.shape(tensor)[0]

            tensor = tf.reshape(tensor, (batch, -1, self.h, self.depth))
            return tf.transpose(tensor, perm=[0, 2, 1, 3])

        Q = split(Q)
        K = split(K)
        V = split(V)
        sdp_a, att_w = sdp_attention(Q, K, V, mask)
        swap = tf.transpose(sdp_a, perm=[0, 2, 1, 3])
        merged = tf.reshape(swap, (batch, -1, self.dm))
        Y = self.linear(merged)
        return Y, att_w


class EncoderBlock(tf.keras.layers.Layer):
    """Transformer encoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize instance."""
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Functionality of the layer."""
        mha_out, _ = self.mha(x, x, x, mask)
        norm1 = self.layernorm1(self.dropout1(mha_out, training=training) + x,
                                training=training)
        y1 = self.dense_hidden(norm1, training=training)
        y2 = self.dense_output(y1, training=training)

        norm2 = self.layernorm2(self.dropout2(y2, training=training) + norm1,
                                training=training)
        return norm2


class DecoderBlock(tf.keras.layers.Layer):
    """Transformer decoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize instance."""
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Functionality of the layer."""
        mha1, _ = self.mha1(x, x, x, look_ahead_mask)

        norm1 = self.layernorm1(self.dropout1(mha1, training=training) + x,
                                training=training)
        mha2, _ = self.mha2(norm1, encoder_output,
                            encoder_output, padding_mask)
        norm2 = self.layernorm2(self.dropout2(mha2, training=training) + norm1,
                                training=training)
        y1 = self.dense_hidden(norm2, training=training)
        y2 = self.dense_output(y1, training=training)

        norm3 = self.layernorm3(self.dropout3(y2, training=training) + norm2,
                                training=training)
        return norm3


class Encoder(tf.keras.layers.Layer):
    """Transformer encoder stack."""

    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """Initialize class."""
        super().__init__()
        self.N = N
        self.dm = dm

        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """Call the layer."""
        input_len = x.shape[1]
        p_e = self.positional_encoding[:input_len]
        x = self.embedding(x)

        # This is the reason we scale it:
        # https://stackoverflow.com/questions/56930821/why-does-embedding-vector-multiplied-by-a-constant-in-transformer-model
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        x = x + p_e
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training, mask)
        return x


class Decoder(tf.keras.layers.Layer):
    """Transformer encoder stack."""

    def __init__(self, N, dm, h, hidden,
                 target_vocab, max_seq_len, drop_rate=0.1):
        """Initialize class."""
        super().__init__()
        self.N = N
        self.dm = dm

        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """Call the layer."""
        input_len = x.shape[1]
        p_e = self.positional_encoding[:input_len]
        x = self.embedding(x)

        # This is the reason we scale it:
        # https://stackoverflow.com/questions/56930821/why-does-embedding-vector-multiplied-by-a-constant-in-transformer-model
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        x = x + p_e
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, encoder_output, training, look_ahead_mask,
                      padding_mask)
        return x


class Transformer(tf.keras.Model):
    """
    Transformer model implimentation in keras.

    N - the number of blocks in the encoder and decoder
    dm - the dimensionality of the model
    h - the number of heads
    hidden - the number of hidden units in the fully connected layers
    input_vocab - the size of the input vocabulary
    target_vocab - the size of the target vocabulary
    max_seq_input - the maximum sequence length possible for the input
    max_seq_target - the maximum sequence length possible for the target
    drop_rate - the dropout rate
    """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """Initialize Model."""
        super().__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def create_masks(self, inputs, target):
        """Create masks for the transformer training."""
        l_out = tf.shape(target)[1]

        padding_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]

        look_ahead = 1 - tf.linalg.band_part(tf.ones((l_out, l_out)), -1, 0)

        target_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
        target_mask = target_mask[:, tf.newaxis, tf.newaxis, :]

        combined_mask = tf.maximum(target_mask, look_ahead)

        return padding_mask, combined_mask, padding_mask

    def train_step(self, data):
        """Train step."""
        target, _ = data
        target_real = target[:, 1:]

        with tf.GradientTape() as tape:
            y_pred = self(data, training=True)
            loss = self.compiled_loss(target_real, y_pred)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(target_real, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training):
        """Call Model."""
        # This is the context of the input sequence
        # (the relation between each word)
        # devived by self attention
        target, x = inputs
        target_inp = target[:, :-1]
        (encoder_mask, look_ahead_mask,
            decoder_mask) = self.create_masks(x, target_inp)
        y_encoder = self.encoder(x, training, encoder_mask)
        y_decoder = self.decoder(target_inp, y_encoder, training,
                                 look_ahead_mask, decoder_mask)

        Y = self.linear(y_decoder)

        return Y
