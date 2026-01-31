#!/usr/bin/env python3
"""데이터셋 테스트 스크립트"""

import sys
sys.path.insert(0, '/home/work/skku/icip/EEG_MVSelect')

from src.datasets import EEGConnectivityDataset

print("="*70)
print("EEG Connectivity Dataset Test")
print("="*70)

# 데이터셋 로드
train_dataset = EEGConnectivityDataset(
    root_dir='/home/work/skku/icip/FCM_Images_HERMES_v2',
    split='train'
)

val_dataset = EEGConnectivityDataset(
    root_dir='/home/work/skku/icip/FCM_Images_HERMES_v2',
    split='val'
)

test_dataset = EEGConnectivityDataset(
    root_dir='/home/work/skku/icip/FCM_Images_HERMES_v2',
    split='test'
)

print("\n" + "="*70)
print("Dataset Statistics")
print("="*70)
print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Total subjects: {train_dataset.num_class}")
print(f"Views per sample: {train_dataset.num_views}")

# 샘플 테스트
print("\n" + "="*70)
print("Sample Test")
print("="*70)

if len(train_dataset) > 0:
    imgs, label, keep_cams = train_dataset[0]
    print(f"Sample 0:")
    print(f"  Images shape: {imgs.shape}")
    print(f"  Label: {label} (Class: {train_dataset.classes[label]})")
    print(f"  Keep_cams shape: {keep_cams.shape}")
    
    print("\nView names:")
    for i in range(train_dataset.num_views):
        print(f"  View {i:2d}: {train_dataset.get_view_name(i)}")
else:
    print("Warning: No samples in training set!")

print("\n" + "="*70)
print("Test Completed Successfully!")
print("="*70)
