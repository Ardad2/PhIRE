#!/usr/bin/env python3
"""
Verify PhIRE-compatible TFRecords
"""

import tensorflow as tf
import numpy as np
from pathlib import Path

tfrecord_dir = Path("/home/adadhwal/PhIRE/wtk_tfrecords")

print("="*80)
print("TFRecord Verification")
print("="*80)

for tfr_file in sorted(tfrecord_dir.glob("*.tfrecord")):
    print(f"\n📄 {tfr_file.name}")
    
    # Read first record
    dataset = tf.data.TFRecordDataset(str(tfr_file))
    
    for raw_record in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        # Extract features
        features = example.features.feature
        index = features['index'].int64_list.value[0]
        
        lr_shape = (
            features['lr_shape_0'].int64_list.value[0],
            features['lr_shape_1'].int64_list.value[0],
            features['lr_shape_2'].int64_list.value[0]
        )
        
        hr_shape = (
            features['hr_shape_0'].int64_list.value[0],
            features['hr_shape_1'].int64_list.value[0],
            features['hr_shape_2'].int64_list.value[0]
        )
        
        # Decode arrays
        lr_bytes = features['lr_data'].bytes_list.value[0]
        hr_bytes = features['hr_data'].bytes_list.value[0]
        
        lr_data = np.frombuffer(lr_bytes, dtype=np.float32).reshape(lr_shape)
        hr_data = np.frombuffer(hr_bytes, dtype=np.float32).reshape(hr_shape)
        
        print(f"  Sample {index}:")
        print(f"    LR shape: {lr_shape} (expect (?, ?, 2))")
        print(f"    HR shape: {hr_shape} (expect (?, ?, 2))")
        print(f"    LR u range: [{lr_data[:,:,0].min():.2f}, {lr_data[:,:,0].max():.2f}]")
        print(f"    LR v range: [{lr_data[:,:,1].min():.2f}, {lr_data[:,:,1].max():.2f}]")
        print(f"    HR u range: [{hr_data[:,:,0].min():.2f}, {hr_data[:,:,0].max():.2f}]")
        print(f"    HR v range: [{hr_data[:,:,1].min():.2f}, {hr_data[:,:,1].max():.2f}]")
        
        # Check it's vector wind (2 channels)
        assert lr_shape[2] == 2, "LR should have 2 channels!"
        assert hr_shape[2] == 2, "HR should have 2 channels!"
        print(f"    ✅ Correct format: (H, W, 2)")

print("\n✅ All TFRecords verified!")
