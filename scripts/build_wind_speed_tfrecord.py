#!/usr/bin/env python3
"""Build a scalar-speed TFRecord from an existing vector wind TFRecord.

This script reads paired wind records with channels [u, v] and writes a new
paired TFRecord whose LR/HR tensors contain scalar speed magnitude
sqrt(u^2 + v^2) as a single channel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def _parse_vector_record(serialized):
    feats = {
        'index': tf.io.FixedLenFeature([], tf.int64),
        'data_LR': tf.io.FixedLenFeature([], tf.string),
        'h_LR': tf.io.FixedLenFeature([], tf.int64),
        'w_LR': tf.io.FixedLenFeature([], tf.int64),
        'data_HR': tf.io.FixedLenFeature([], tf.string),
        'h_HR': tf.io.FixedLenFeature([], tf.int64),
        'w_HR': tf.io.FixedLenFeature([], tf.int64),
        'c': tf.io.FixedLenFeature([], tf.int64),
    }
    ex = tf.io.parse_single_example(serialized, feats)

    h_lr = tf.cast(ex['h_LR'], tf.int32)
    w_lr = tf.cast(ex['w_LR'], tf.int32)
    h_hr = tf.cast(ex['h_HR'], tf.int32)
    w_hr = tf.cast(ex['w_HR'], tf.int32)
    c = tf.cast(ex['c'], tf.int32)

    data_lr = tf.io.decode_raw(ex['data_LR'], tf.float64)
    data_hr = tf.io.decode_raw(ex['data_HR'], tf.float64)
    data_lr = tf.reshape(data_lr, [h_lr, w_lr, c])
    data_hr = tf.reshape(data_hr, [h_hr, w_hr, c])
    return ex['index'], data_lr, data_hr


def _speed_from_uv(arr):
    arr = np.asarray(arr)
    if arr.ndim != 3 or arr.shape[-1] < 2:
        raise ValueError(f'Expected [H,W,C>=2] vector field, got {arr.shape}')
    speed = np.sqrt(np.square(arr[..., 0]) + np.square(arr[..., 1]))
    return speed[..., None]


def _write_scalar_example(writer, idx, lr_speed, hr_speed):
    h_lr, w_lr, c_lr = lr_speed.shape
    h_hr, w_hr, c_hr = hr_speed.shape
    if c_lr != 1 or c_hr != 1:
        raise ValueError('Scalar speed tensors must have one channel')

    features = tf.train.Features(feature={
        'index': _int64_feature(idx),
        'data_LR': _bytes_feature(np.asarray(lr_speed, dtype=np.float64).tobytes()),
        'h_LR': _int64_feature(h_lr),
        'w_LR': _int64_feature(w_lr),
        'data_HR': _bytes_feature(np.asarray(hr_speed, dtype=np.float64).tobytes()),
        'h_HR': _int64_feature(h_hr),
        'w_HR': _int64_feature(w_hr),
        'c': _int64_feature(1),
    })
    writer.write(tf.train.Example(features=features).SerializeToString())


def main():
    ap = argparse.ArgumentParser(description='Build scalar speed TFRecord from vector wind TFRecord')
    ap.add_argument('--input', required=True, help='Input vector wind TFRecord')
    ap.add_argument('--output', required=True, help='Output scalar-speed TFRecord')
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tf.compat.v1.disable_eager_execution()

    raw_ph = tf.compat.v1.placeholder(tf.string, shape=[])
    idx_t, lr_t, hr_t = _parse_vector_record(raw_ph)

    count = 0
    with tf.compat.v1.Session() as sess, tf.io.TFRecordWriter(str(out_path)) as writer:
        for raw in tf.compat.v1.io.tf_record_iterator(args.input):
            idx, lr, hr = sess.run([idx_t, lr_t, hr_t], feed_dict={raw_ph: raw})
            lr_speed = _speed_from_uv(lr)
            hr_speed = _speed_from_uv(hr)
            _write_scalar_example(writer, idx, lr_speed, hr_speed)
            count += 1

    print(f'Wrote {count} records to {out_path}')


if __name__ == '__main__':
    main()
