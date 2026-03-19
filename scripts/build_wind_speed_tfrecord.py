#!/usr/bin/env python3
"""
Build a scalar-speed TFRecord from an existing wind MR-HR TFRecord.

Input records are expected to contain vector wind channels [u, v].
Output records contain 1-channel speed magnitude sqrt(u^2 + v^2).

This keeps the current vector pipeline untouched and creates a parallel
scalar-only training/testing dataset for the smallest possible experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


def _parse_vector_record(example_proto):
    feats = {
        "index": tf.FixedLenFeature([], tf.int64),
        "data_LR": tf.FixedLenFeature([], tf.string),
        "h_LR": tf.FixedLenFeature([], tf.int64),
        "w_LR": tf.FixedLenFeature([], tf.int64),
        "data_HR": tf.FixedLenFeature([], tf.string),
        "h_HR": tf.FixedLenFeature([], tf.int64),
        "w_HR": tf.FixedLenFeature([], tf.int64),
        "c": tf.FixedLenFeature([], tf.int64),
    }
    ex = tf.parse_single_example(example_proto, feats)

    h_lr = tf.cast(ex["h_LR"], tf.int32)
    w_lr = tf.cast(ex["w_LR"], tf.int32)
    h_hr = tf.cast(ex["h_HR"], tf.int32)
    w_hr = tf.cast(ex["w_HR"], tf.int32)
    c = tf.cast(ex["c"], tf.int32)

    lr = tf.decode_raw(ex["data_LR"], tf.float64)
    hr = tf.decode_raw(ex["data_HR"], tf.float64)

    lr = tf.reshape(lr, [h_lr, w_lr, c])
    hr = tf.reshape(hr, [h_hr, w_hr, c])

    return ex["index"], lr, hr


def _speed_from_uv(arr):
    arr = np.asarray(arr)
    if arr.ndim != 3 or arr.shape[-1] < 2:
        raise ValueError(f"Expected [H,W,C>=2], got {arr.shape}")
    speed = np.sqrt(np.square(arr[..., 0]) + np.square(arr[..., 1]))
    return speed[..., None]


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def write_scalar_record(writer, idx, lr_speed, hr_speed):
    h_lr, w_lr, c_lr = lr_speed.shape
    h_hr, w_hr, c_hr = hr_speed.shape
    if c_lr != 1 or c_hr != 1:
        raise ValueError("Scalar speed arrays must have channel count 1")

    feats = tf.train.Features(
        feature={
            "index": _int64_feature(idx),
            "data_LR": _bytes_feature(np.asarray(lr_speed, dtype=np.float64).tobytes()),
            "h_LR": _int64_feature(h_lr),
            "w_LR": _int64_feature(w_lr),
            "data_HR": _bytes_feature(np.asarray(hr_speed, dtype=np.float64).tobytes()),
            "h_HR": _int64_feature(h_hr),
            "w_HR": _int64_feature(w_hr),
            "c": _int64_feature(1),
        }
    )
    writer.write(tf.train.Example(features=feats).SerializeToString())


def main():
    ap = argparse.ArgumentParser(description="Build scalar-speed TFRecord from vector wind TFRecord")
    ap.add_argument("--input", required=True, help="Input wind MR-HR TFRecord")
    ap.add_argument("--output", required=True, help="Output scalar-speed TFRecord")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = tf.data.TFRecordDataset([args.input])
    it = ds.make_one_shot_iterator().get_next()

    n = 0
    with tf.Session() as sess, tf.python_io.TFRecordWriter(str(out_path)) as writer:
        while True:
            try:
                raw = sess.run(it)
                idx_t, lr_t, hr_t = _parse_vector_record(tf.convert_to_tensor(raw))
                idx, lr, hr = sess.run([idx_t, lr_t, hr_t])

                lr_speed = _speed_from_uv(lr)
                hr_speed = _speed_from_uv(hr)

                write_scalar_record(writer, idx, lr_speed, hr_speed)
                n += 1
            except tf.errors.OutOfRangeError:
                break

    print(f"Wrote {n} records to {out_path}")


if __name__ == "__main__":
    main()
