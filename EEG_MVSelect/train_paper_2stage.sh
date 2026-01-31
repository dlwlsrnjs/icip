#!/bin/bash

# Neuroinformatics 2024 논문 철학 따른 모델 2-stage 학습
# Stage 1: 모든 view 사용하여 base network 학습
# Stage 2: MVSelect로 3개 view 선택하며 fine-tuning

LOG_DIR="logs/paper_exact_2stage"
mkdir -p ${LOG_DIR}

echo "========================================"
echo "Paper-Inspired Att-CNN 2-Stage Training"
echo "========================================"
echo ""
echo "논문 철학:"
echo "  - Input: 224×224 connectivity heatmap"
echo "  - Architecture: 3 Conv layers (32→64→128)"
echo "  - Channel Attention (reduction=8)"
echo "  - Multi-view aggregation with MVSelect"
echo ""

# Stage 1: Base network training (모든 15 views 사용)
echo "=== Stage 1: Base Network Training ==="
echo "Training with all 15 views..."

python main.py \
    --use_paper_exact \
    --data_root /home/work/skku/icip/FCM_Images_HERMES_v2 \
    --image_size 224 \
    --batch_size 16 \
    --epochs 80 \
    --lr 1e-3 \
    --weight_decay 5e-4 \
    --steps 0 \
    --save_dir ${LOG_DIR}/stage1 \
    --seed 42

echo ""
echo "Stage 1 completed! Checkpoint saved to ${LOG_DIR}/stage1"
echo ""

# Stage 2: MVSelect fine-tuning (3 views 선택)
echo "=== Stage 2: MVSelect Fine-tuning ==="
echo "Fine-tuning with view selection (3 views)..."

# Stage 1의 best model 찾기
STAGE1_BEST=$(find ${LOG_DIR}/stage1 -name "best_model.pth" 2>/dev/null | head -1)

if [ -z "$STAGE1_BEST" ]; then
    echo "Error: Stage 1 checkpoint not found!"
    exit 1
fi

echo "Loading checkpoint: ${STAGE1_BEST}"

python main.py \
    --use_paper_exact \
    --data_root /home/work/skku/icip/FCM_Images_HERMES_v2 \
    --image_size 224 \
    --batch_size 16 \
    --epochs 60 \
    --lr 5e-4 \
    --base_lr_ratio 0.1 \
    --weight_decay 5e-4 \
    --steps 3 \
    --rl_loss_weight 0.5 \
    --resume ${STAGE1_BEST} \
    --save_dir ${LOG_DIR}/stage2 \
    --seed 42

echo ""
echo "========================================"
echo "2-Stage Training Completed!"
echo "========================================"
echo "Stage 1 results: ${LOG_DIR}/stage1"
echo "Stage 2 results: ${LOG_DIR}/stage2"
echo ""
echo "논문 철학 적용:"
echo "  ✓ 224×224 connectivity heatmap 사용"
echo "  ✓ 3-layer CNN with Channel Attention"
echo "  ✓ Multi-view selection (15→3 views)"
echo "  ✓ ~360K parameters (lightweight)"
