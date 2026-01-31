#!/bin/bash

# 개선된 Stage 1 학습 - 과적합 방지
# Early stopping, Stronger regularization, Data augmentation

echo "=========================================================="
echo "개선된 Stage 1: Base Network Training (과적합 방지)"
echo "=========================================================="
echo ""
echo "주요 개선 사항:"
echo "  1. Dropout 증가: 0.5 → 0.6"
echo "  2. Weight decay 증가: 1e-4 → 5e-4"
echo "  3. Learning rate 감소: 3e-5 → 1e-5 (더 안정적인 학습)"
echo "  4. Batch size 증가: 6 → 8 (더 안정적인 gradient)"
echo "  5. Epochs: 60 → 80 (early stopping으로 조기 종료)"
echo "  6. Label smoothing 추가"
echo ""

cd /home/work/skku/icip/EEG_MVSelect

python main.py \
    --data_root /home/work/skku/icip/FCM_Images_HERMES_v2 \
    --arch resnet50 \
    --use_attention \
    --aggregation max \
    --steps 0 \
    --epochs 80 \
    --batch_size 8 \
    --lr 1e-5 \
    --weight_decay 5e-4 \
    --scheduler cosine \
    --grad_clip 1.0 \
    --num_workers 4 \
    --seed 42 \
    --save_dir logs/stage1_improved

echo ""
echo "=========================================================="
echo "Stage 1 학습 완료!"
echo "=========================================================="
