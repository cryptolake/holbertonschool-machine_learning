#!/usr/bin/env python3
"""Creating the model."""
import tensorflow as tf


def create_model(batches):
    """Create model."""
    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(batches, return_sequences=False),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
        ])
    return lstm_model


def compile_and_fit(model, train, val, epochs=20, patience=2):
    """Compile and fit model."""
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

    model.fit(train, epochs=epochs,
              validation_data=val,
              callbacks=[early_stopping])


def main():
    """Run main function."""
    lstm_model = create_model(32)
    train_dataset, val_dataset = (tf.data.Dataset.load('./datasets/train'),
                                  tf.data.Dataset.load('./datasets/val'))
    compile_and_fit(lstm_model, train_dataset, val_dataset)
    lstm_model.save('Bitcoin_model')


if __name__ == "__main__":
    main()
