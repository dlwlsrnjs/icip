#!/bin/bash

# EEG-MVSelect 학습 스크립트

# Step 1: 기본 분류 네트워크 학습 (모든 뷰 사용)
echo "=========================================="
echo "Step 1: Training Base Classification Network"
echo "=========================================="

python main.py \
    --arch resnet18 \
    --aggregation max \
    --epochs 50 \
    --batch_size 8 \
    --lr 5e-5 \
    --steps 0 \
    --scheduler cosine \
    --seed 42

# Step 2: MVSelect 모듈 학습
echo ""
echo "=========================================="
echo "Step 2: Training MVSelect Module"
echo "=========================================="

python main.py \
    --arch resnet18 \
    --aggregation max \
    --epochs 30 \
    --batch_size 8 \
    --lr 5e-5 \
    --select_lr 1e-4 \
    --base_lr_ratio 0.1 \
    --steps 2 \
    --init_cam random \
    --rl_loss_weight 0.1 \
    --scheduler cosine \
    --seed 42

echo ""
echo "Training completed!"
