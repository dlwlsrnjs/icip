#!/bin/bash

# 논문 스타일 경량 모델 2-Stage 학습
# Stage 1: Base network (모든 뷰)
# Stage 2: MVSelect (3-view selection)

echo "=========================================================="
echo "논문 스타일 경량 Attention 모델 2-Stage 학습"
echo "=========================================================="
echo ""
echo "모델 특징:"
echo "  - 3-layer CNN (논문과 동일)"
echo "  - Channel Attention만 (CBAM의 절반)"
echo "  - ~1M parameters (ResNet50의 1/25)"
echo "  - 과적합 방지를 위한 단순 구조"
echo ""

cd /home/work/skku/icip/EEG_MVSelect

# Stage 1: Base Network 학습
echo "=========================================================="
echo "[Stage 1] Base Network Training"
echo "=========================================================="
echo "  - Architecture: 3-layer CNN + Channel Attention"
echo "  - Views: 15 (all views)"
echo "  - Epochs: 80 (early stopping)"
echo "  - Batch size: 16"
echo "  - Learning rate: 1e-3"
echo ""

python main.py \
    --data_root /home/work/skku/icip/FCM_Images_HERMES_v2 \
    --use_lightweight \
    --aggregation max \
    --steps 0 \
    --epochs 80 \
    --batch_size 16 \
    --lr 1e-3 \
    --weight_decay 5e-4 \
    --scheduler cosine \
    --grad_clip 1.0 \
    --num_workers 4 \
    --seed 42 \
    --save_dir logs/lightweight_stage1

# 최고 성능 체크포인트 찾기
BEST_CKPT=$(ls -t logs/lightweight_stage1/*/best_model.pth 2>/dev/null | head -1)

if [ -z "$BEST_CKPT" ]; then
    echo ""
    echo "Error: Stage 1 체크포인트를 찾을 수 없습니다!"
    exit 1
fi

echo ""
echo "=========================================================="
echo "Stage 1 완료!"
echo "=========================================================="
echo "Best checkpoint: $BEST_CKPT"
echo ""
sleep 3

# Stage 2: MVSelect Fine-tuning
echo "=========================================================="
echo "[Stage 2] MVSelect Fine-tuning"
echo "=========================================================="
echo "  - View Selection: 3 views (from 15)"
echo "  - Epochs: 60"
echo "  - Batch size: 16"
echo "  - Base LR: 1e-3 (ratio=0.1 → 1e-4)"
echo "  - Select LR: 5e-4"
echo "  - RL loss weight: 0.5"
echo ""

python main.py \
    --data_root /home/work/skku/icip/FCM_Images_HERMES_v2 \
    --use_lightweight \
    --aggregation max \
    --steps 3 \
    --epochs 60 \
    --batch_size 16 \
    --lr 1e-3 \
    --select_lr 5e-4 \
    --base_lr_ratio 0.1 \
    --weight_decay 5e-4 \
    --scheduler cosine \
    --grad_clip 1.0 \
    --rl_loss_weight 0.5 \
    --num_workers 4 \
    --seed 42 \
    --resume $BEST_CKPT \
    --save_dir logs/lightweight_stage2

echo ""
echo "=========================================================="
echo "2-Stage 학습 완료!"
echo "=========================================================="
echo ""
echo "결과 확인:"
echo "  Stage 1 (Base): logs/lightweight_stage1/"
echo "  Stage 2 (MVSelect): logs/lightweight_stage2/"
