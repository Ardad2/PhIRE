#!/usr/bin/env python3
"""
Test if TFRecords load through PhIRE's actual parser
"""

import tensorflow as tf
from pathlib import Path

# PhIRE's EXACT parser (from utils.py)
def _parse_test_(example_proto):
    """PhIRE's test parser"""
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
    
    h_lr = parsed['h_LR']
    w_lr = parsed['w_LR']
    h_hr = parsed['h_HR']
    w_hr = parsed['w_HR']
    c = parsed['c']
    
    # Decode as float64 (PhIRE standard)
    data_LR = tf.io.decode_raw(parsed['data_LR'], tf.float64)
    data_HR = tf.io.decode_raw(parsed['data_HR'], tf.float64)
    
    # Reshape
    data_LR = tf.reshape(data_LR, [h_lr, w_lr, c])
    data_HR = tf.reshape(data_HR, [h_hr, w_hr, c])
    
    return data_LR, data_HR, parsed['index']


tfrecord_dir = Path("/home/adadhwal/PhIRE/wtk_tfrecords_native")

print("="*80)
print("Testing TFRecords with PhIRE's Parser")
print("="*80)

for tfr_file in sorted(tfrecord_dir.glob("*_train.tfrecord")):
    print(f"\n📄 {tfr_file.name}")
    
    try:
        dataset = tf.data.TFRecordDataset(str(tfr_file))
        dataset = dataset.map(_parse_test_)
        
        # Try to load first record
        for data_lr, data_hr, index in dataset.take(1):
            print(f"  ✅ Parsed successfully!")
            print(f"     LR shape: {data_lr.shape}")
            print(f"     HR shape: {data_hr.shape}")
            print(f"     LR dtype: {data_lr.dtype}")
            print(f"     HR dtype: {data_hr.dtype}")
            print(f"     Index: {index.numpy()}")
            
            # Check values
            print(f"     LR range: [{data_lr.numpy().min():.2f}, {data_lr.numpy().max():.2f}]")
            print(f"     HR range: [{data_hr.numpy().min():.2f}, {data_hr.numpy().max():.2f}]")
            
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

print("\n✅ All parseable TFRecords verified!")
