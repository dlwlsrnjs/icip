#!/bin/bash

# EEG-MVSelect 평가 스크립트

if [ -z "$1" ]; then
    echo "Usage: ./evaluate.sh <checkpoint_path>"
    echo "Example: ./evaluate.sh logs/resnet18_max_steps2_lr5e-05_bs8_20260130_120000/best_model.pth"
    exit 1
fi

CHECKPOINT=$1

echo "=========================================="
echo "Evaluating EEG-MVSelect Model"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo ""

python main.py \
    --arch resnet18 \
    --aggregation max \
    --steps 2 \
    --batch_size 8 \
    --eval \
    --resume $CHECKPOINT

echo ""
echo "Evaluation completed!"
