#!/bin/bash

# Attention-Enhanced EEG MVSelect 2-Stage 학습
# ResNet50 + CBAM + Multi-View Attention + View Selection Attention

echo "=========================================================="
echo "Attention-Enhanced EEG MVSelect 2-Stage Training"
echo "=========================================================="
echo ""
echo "모델 개선 사항:"
echo "  1. ResNet18 → ResNet50 (더 깊은 네트워크)"
echo "  2. CBAM: Channel + Spatial Attention"
echo "  3. Multi-View Attention: 뷰 간 관계 학습 (8 heads)"
echo "  4. View Selection Attention: 더 나은 뷰 선택"
echo ""
echo "학습 개선 사항:"
echo "  1. RL loss weight: 0.1 → 0.5"
echo "  2. Select LR: 1e-4 → 5e-4"
echo "  3. Base network는 천천히 학습 (lr_ratio=0.1)"
echo ""

# Stage 1: Base Network 학습 (60 epochs, 모든 뷰)
echo "=========================================================="
echo "[Stage 1] Base Network Training"
echo "=========================================================="
echo "  - Architecture: ResNet50 + CBAM + MV Attention"
echo "  - Views: 15 (all views)"
echo "  - Epochs: 60"
echo "  - Batch size: 6 (ResNet50은 메모리 많이 사용)"
echo "  - Learning rate: 3e-5"
echo ""

python main.py \
    --data_root /home/work/skku/icip/FCM_Images_HERMES_v2 \
    --arch resnet50 \
    --use_attention \
    --aggregation max \
    --steps 0 \
    --epochs 60 \
    --batch_size 6 \
    --lr 3e-5 \
    --weight_decay 1e-4 \
    --scheduler cosine \
    --grad_clip 1.0 \
    --num_workers 4 \
    --seed 42 \
    --save_dir logs/attention_stage1_base

# 최고 성능 체크포인트 찾기
BEST_CKPT=$(ls -t logs/attention_stage1_base/*/best_model.pth 2>/dev/null | head -1)

if [ -z "$BEST_CKPT" ]; then
    echo ""
    echo "Error: Stage 1 체크포인트를 찾을 수 없습니다!"
    echo "학습이 제대로 완료되었는지 확인해주세요."
    exit 1
fi

echo ""
echo "=========================================================="
echo "Stage 1 완료!"
echo "=========================================================="
echo "Best checkpoint: $BEST_CKPT"
echo ""
sleep 3

# Stage 2: MVSelect Fine-tuning (60 epochs, 3-view selection)
echo "=========================================================="
echo "[Stage 2] MVSelect Fine-tuning"
echo "=========================================================="
echo "  - Architecture: ResNet50 + All Attentions"
echo "  - View Selection: 3 views (from 15)"
echo "  - Epochs: 60"
echo "  - Batch size: 6"
echo "  - Base LR: 3e-5 (ratio=0.1 → 실제 3e-6)"
echo "  - Select LR: 5e-4 (RL agent 빠른 학습)"
echo "  - RL loss weight: 0.5 (강화)"
echo ""

python main.py \
    --data_root /home/work/skku/icip/FCM_Images_HERMES_v2 \
    --arch resnet50 \
    --use_attention \
    --aggregation max \
    --steps 3 \
    --epochs 60 \
    --batch_size 6 \
    --lr 3e-5 \
    --select_lr 5e-4 \
    --base_lr_ratio 0.1 \
    --weight_decay 1e-4 \
    --scheduler cosine \
    --grad_clip 1.0 \
    --rl_loss_weight 0.5 \
    --num_workers 4 \
    --seed 42 \
    --resume $BEST_CKPT \
    --save_dir logs/attention_stage2_mvselect

echo ""
echo "=========================================================="
echo "2-Stage Training 완료!"
echo "=========================================================="
echo ""
echo "결과 확인:"
echo "  Stage 1 (Base): logs/attention_stage1_base/"
echo "  Stage 2 (MVSelect): logs/attention_stage2_mvselect/"
echo ""
echo "성능 비교를 위해 로그 파일을 확인하세요."
