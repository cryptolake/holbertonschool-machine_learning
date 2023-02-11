#!/usr/bin/env python3
"""Train transformer."""
import tensorflow as tf

Dataset = __import__('3-dataset').Dataset
Transformer = __import__('5-transformer').Transformer


# https://www.tensorflow.org/text/tutorials/transformer#set_up_the_optimizer
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Scheduler for the learning in accordance to the attention paper."""

    def __init__(self, d_model, warmup_steps=4000):
        """Initialize the class."""
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """Learning rate scheduler."""
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


# https://www.tensorflow.org/text/tutorials/transformer#set_up_the_loss_and_metrics

def masked_loss(label, pred):
    """
    Calculate the loss without padding.

    Since the target sequences are padded,
    it is important to apply a padding mask when calculating the loss.
    """
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    """
    Calculate the accuracy without padding.

    Since the target sequences are padded,
    it is important to apply a padding mask when calculating the accuracy.
    """
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)


class CustomLogger(tf.keras.callbacks.Callback):
    """Custom logger to print."""

    def __init__(self, b_display):
        """Initialize class."""
        self.b_display = b_display
        self._epoch = 1

    def on_batch_end(self, batch, logs={}):
        """Perform on batch end."""
        loss = logs['loss']
        acc = logs["masked_accuracy"]
        if batch % self.b_display == 0:
            print('Epoch {}, batch {}: loss {} accuracy {}'.
                  format(self._epoch, batch, loss, acc))

    def on_epoch_end(self, epoch, logs={}):
        """Perform on epoch end."""
        loss = logs["loss"]
        acc = logs["masked_accuracy"]
        self._epoch = epoch
        print('Epoch {}: loss {} accuracy {}'
              .format(epoch, loss, acc))


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """Train the transformer."""
    ds = Dataset(batch_size, max_len)
    target_size = ds.tokenizer_pt.vocab_size + 2
    input_size = ds.tokenizer_en.vocab_size + 2

    transformer = Transformer(N, dm, h, hidden, input_size,
                              target_size, max_len, max_len)

    lr_scheduler = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(lr_scheduler, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    out_batch = CustomLogger(50)
    transformer.compile(
            loss=masked_loss,
            optimizer=optimizer,
            metrics=[masked_accuracy])

    transformer.fit(ds.data_train,
                    epochs=epochs,
                    validation_data=ds.data_valid,
                    steps_per_epoch=ds.data_train_size // batch_size,
                    validation_steps=ds.data_valid_size // batch_size,
                    callbacks=[out_batch],
                    verbose=False)
