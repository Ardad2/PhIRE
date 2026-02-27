#!/usr/bin/env python3
"""
Verify the clone matches example_data behavior
"""

import tensorflow as tf
from pathlib import Path

def _parse_test_(example_proto):
    """PhIRE's parser"""
    features = {
        'data_LR': tf.io.FixedLenFeature([], tf.string),
        'h_LR': tf.io.FixedLenFeature([], tf.int64),
        'w_LR': tf.io.FixedLenFeature([], tf.int64),
        'data_HR': tf.io.FixedLenFeature([], tf.string),
        'h_HR': tf.io.FixedLenFeature([], tf.int64),
        'w_HR': tf.io.FixedLenFeature([], tf.int64),
        'c': tf.io.FixedLenFeature([], tf.int64),
        'index': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, features)
    
    data_LR = tf.io.decode_raw(parsed['data_LR'], tf.float64)
    data_HR = tf.io.decode_raw(parsed['data_HR'], tf.float64)
    
    data_LR = tf.reshape(data_LR, [parsed['h_LR'], parsed['w_LR'], parsed['c']])
    data_HR = tf.reshape(data_HR, [parsed['h_HR'], parsed['w_HR'], parsed['c']])
    
    return data_LR, data_HR, parsed['index']

clone_dir = Path("/home/adadhwal/PhIRE/example_data_clone")

print("="*80)
print("Clone Verification")
print("="*80)

for tfr_file in sorted(clone_dir.glob("wind_*.tfrecord")):
    print(f"\n📄 {tfr_file.name}")
    
    dataset = tf.data.TFRecordDataset(str(tfr_file))
    dataset = dataset.map(_parse_test_)
    
    records = list(dataset)
    n_records = len(records)
    
    print(f"  Records: {n_records} (expect 5)")
    
    if n_records == 5:
        print(f"  ✅ Correct sample count")
    else:
        print(f"  ❌ Wrong sample count!")
    
    # Check first record
    data_lr, data_hr, index = records[0]
    print(f"\n  Sample 0:")
    print(f"    LR: {data_lr.shape} (dtype: {data_lr.dtype})")
    print(f"    HR: {data_hr.shape} (dtype: {data_hr.dtype})")
    
    if data_hr.shape[0] == 160 and data_hr.shape[1] == 160:
        print(f"    ✅ HR is 160×160 (matches --patch 160)")
    else:
        print(f"    ❌ HR is NOT 160×160!")
    
    if data_hr.shape[2] == 2:
        print(f"    ✅ Has 2 channels [u, v]")
    else:
        print(f"    ❌ Wrong channel count!")
    
    if data_lr.dtype == tf.float64:
        print(f"    ✅ dtype is float64")
    else:
        print(f"    ❌ Wrong dtype!")

print("\n" + "="*80)
print("Clone Status:")
print("  If all checks pass, this is a drop-in replacement!")
print("="*80)
