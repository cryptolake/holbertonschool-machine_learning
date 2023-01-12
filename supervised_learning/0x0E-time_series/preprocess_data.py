#!/usr/bin/env python3
"""Preprocessing the data."""
import pandas as pd
import numpy as np
import tensorflow as tf


def clean_transform_data(df):
    """Clean and transform data."""
    clean_na = df.fillna(method='bfill')
    rel_points = clean_na[clean_na['Timestamp'] >= 1.50*1e9]
    useful_feat = rel_points.drop(['Open', 'High', 'Low', 'Volume_(Currency)',
                                   'Weighted_Price', 'Timestamp'], axis=1)
    time_win = useful_feat.groupby(np.arange(len(useful_feat))//60).mean()
    return time_win


def normalize_data(train, val, test):
    """Normalize the different datasets."""
    train_mean = train.mean()
    train_std = train.std()

    train = (train - train_mean) / train_std
    val = (val - train_mean) / train_std
    test = (test - train_mean) / train_std
    return (train_mean, train_std), (train, val, test)


def split_data(df, t_s=0.7, v_s=0.2, test_s=0.1):
    """Split data into different datasets."""
    if not np.isclose(1.0, t_s + v_s + test_s):
        return None, None, None

    n = len(df)
    t_prop = int(n*t_s)
    val_prop = t_prop + int(n*v_s)
    test_prop = val_prop + int(n*test_s)

    train = df[0:t_prop]
    val = df[t_prop:val_prop]
    test = df[val_prop:test_prop]
    return train, val, test


def split_window(batch):
    """Split window to be used for the RNN."""
    inputs = batch[:, :24, :]
    labels = batch[:, 24, 0]
    return inputs, labels


def make_dataset(data):
    """Create the tf.data.dataset."""
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=25,
      sequence_stride=1,
      shuffle=False,
      batch_size=32)

    ds = ds.map(split_window)

    return ds


def main():
    """Run main function."""
    df = pd.read_csv('data/bitstamp.csv')

    print("Cleaning data...")
    clean_df = clean_transform_data(df)
    print("Splitting data...")
    split_df = split_data(clean_df)
    print("Normalizing...")
    norm_val, norm_data = normalize_data(*split_df)
    with open('norm_val', 'w') as f:
        f.write("{}:{}".format(norm_val[0], norm_val[1]))
    print("Making datasets...")
    final_data = list(map(make_dataset, list(norm_data)))
    dataset_names = ['train', 'val', 'test']
    for data, name in zip(final_data, dataset_names):
        data.save('./datasets/{}'.format(name))
        print("Saved dataset at ./datasets/{}".format(name))


if __name__ == "__main__":
    main()
