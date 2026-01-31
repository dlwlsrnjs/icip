#!/bin/bash

# MVSelect 단일 학습 - 개선된 하이퍼파라미터
# View Selection 성능 향상에 초점

echo "======================================"
echo "MVSelect 개선 학습 (Enhanced)"
echo "======================================"
echo ""
echo "주요 개선 사항:"
echo "  1. RL loss weight: 0.1 → 0.5 (view selection 학습 강화)"
echo "  2. Select LR: 1e-4 → 5e-4 (더 빠른 RL 학습)"
echo "  3. Base LR ratio: 1.0 → 0.1 (base network는 천천히, selection은 빠르게)"
echo "  4. Epochs: 20 → 80 (충분한 학습 시간)"
echo "  5. 4-view selection (3개보다 여유있게)"
echo ""

python main.py \
    --data_root /home/work/skku/icip/FCM_Images_HERMES_v2 \
    --arch resnet18 \
    --aggregation max \
    --steps 4 \
    --epochs 80 \
    --batch_size 8 \
    --lr 5e-5 \
    --select_lr 5e-4 \
    --base_lr_ratio 0.1 \
    --weight_decay 1e-4 \
    --scheduler cosine \
    --grad_clip 1.0 \
    --rl_loss_weight 0.5 \
    --num_workers 4 \
    --seed 42 \
    --save_dir logs/mvselect_enhanced

echo ""
echo "======================================"
echo "학습 완료!"
echo "======================================"
