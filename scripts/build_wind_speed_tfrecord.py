#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

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

def _parse_raw_record(raw_record):
    ex = tf.train.Example()
    ex.ParseFromString(raw_record)
    feat = ex.features.feature
    idx  = int(feat["index"].int64_list.value[0])
    h_lr = int(feat["h_LR"].int64_list.value[0])
    w_lr = int(feat["w_LR"].int64_list.value[0])
    h_hr = int(feat["h_HR"].int64_list.value[0])
    w_hr = int(feat["w_HR"].int64_list.value[0])
    c    = int(feat["c"].int64_list.value[0])
    lr = np.frombuffer(feat["data_LR"].bytes_list.value[0], dtype=np.float64).reshape(h_lr, w_lr, c)
    hr = np.frombuffer(feat["data_HR"].bytes_list.value[0], dtype=np.float64).reshape(h_hr, w_hr, c)
    return idx, lr, hr

def _write_scalar_record(writer, idx, lr_speed, hr_speed):
    h_lr, w_lr, _ = lr_speed.shape
    h_hr, w_hr, _ = hr_speed.shape
    ex = tf.train.Example(features=tf.train.Features(feature={
        "index":   _int64_feature(idx),
        "data_LR": _bytes_feature(np.asarray(lr_speed, dtype=np.float64).tobytes()),
        "h_LR":    _int64_feature(h_lr),
        "w_LR":    _int64_feature(w_lr),
        "data_HR": _bytes_feature(np.asarray(hr_speed, dtype=np.float64).tobytes()),
        "h_HR":    _int64_feature(h_hr),
        "w_HR":    _int64_feature(w_hr),
        "c":       _int64_feature(1),
    }))
    writer.write(ex.SerializeToString())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with tf.io.TFRecordWriter(str(out_path)) as writer:
        for raw in tf.compat.v1.io.tf_record_iterator(str(args.input)):
            idx, lr, hr = _parse_raw_record(raw)
            _write_scalar_record(writer, idx, _speed_from_uv(lr), _speed_from_uv(hr))
            n += 1
    print(f"Wrote {n} records to {out_path}")

if __name__ == "__main__":
    main()
