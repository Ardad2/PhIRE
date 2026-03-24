from pathlib import Path
import numpy as np
import tensorflow as tf

SRC = Path("example_data_extension_500/wind_MR-HR.tfrecord")
OUTDIR = Path("example_data_fixed")
OUTDIR.mkdir(parents=True, exist_ok=True)
DST = OUTDIR / "wind_MR-HR.tfrecord"


def _bytes_feature(value: bytes):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value: int):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def block_average_downsample(x: np.ndarray, factor: int) -> np.ndarray:
    h, w = x.shape
    if h % factor != 0 or w % factor != 0:
        raise ValueError(f"Shape {(h, w)} not divisible by factor {factor}")
    return x.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def serialize_record(mr: np.ndarray, hr: np.ndarray, index: int) -> bytes:
    h_lr, w_lr, c = mr.shape
    h_hr, w_hr, _ = hr.shape

    feature = {
        "data_LR": _bytes_feature(mr.astype(np.float64).tobytes()),
        "h_LR": _int64_feature(h_lr),
        "w_LR": _int64_feature(w_lr),
        "data_HR": _bytes_feature(hr.astype(np.float64).tobytes()),
        "h_HR": _int64_feature(h_hr),
        "w_HR": _int64_feature(w_hr),
        "c": _int64_feature(c),
        "index": _int64_feature(index),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def main():
    n = 0
    with tf.io.TFRecordWriter(str(DST)) as writer:
        for raw in tf.compat.v1.io.tf_record_iterator(str(SRC)):
            ex = tf.train.Example()
            ex.ParseFromString(raw)
            f = ex.features.feature

            h_hr = int(f["h_HR"].int64_list.value[0])
            w_hr = int(f["w_HR"].int64_list.value[0])
            c = int(f["c"].int64_list.value[0])
            index = int(f["index"].int64_list.value[0])

            hr = np.frombuffer(
                f["data_HR"].bytes_list.value[0], dtype=np.float64
            ).reshape(h_hr, w_hr, c)

            hr_u = hr[..., 0]
            hr_v = hr[..., 1]

            mr_u = block_average_downsample(hr_u, 5)
            mr_v = block_average_downsample(hr_v, 5)
            mr = np.stack([mr_u, mr_v], axis=-1)

            writer.write(serialize_record(mr, hr, index))
            n += 1

    print(f"Wrote {n} repaired records to {DST}")


if __name__ == "__main__":
    main()
