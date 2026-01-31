#!/bin/bash

# 개선된 설정으로 5-Fold CV 학습
# 변경사항:
# 1. Data augmentation 최소화 (flip/rotation 제거)
# 2. Dropout 감소 (0.5,0.3 -> 0.3,0.2)
# 3. Learning rate 감소 (1e-3 -> 5e-4)
# 4. Weight decay 감소 (5e-4 -> 1e-4)

LOG_DIR="logs/improved_5fold"
mkdir -p ${LOG_DIR}

echo "========================================"
echo "Improved 5-Fold Cross-Validation"
echo "========================================"

for fold in {0..4}
do
    echo "========================================"
    echo "Fold $((fold+1))/5"
    echo "========================================"
    
    python main.py \
        --use_paper_exact \
        --data_root /home/work/skku/icip/FCM_Images_HERMES_v2 \
        --image_size 224 \
        --batch_size 16 \
        --epochs 80 \
        --lr 5e-4 \
        --weight_decay 1e-4 \
        --steps 0 \
        --save_dir ${LOG_DIR}/fold_${fold} \
        --seed 42 \
        --use_cv \
        --cv_fold ${fold} \
        --n_folds 5
    
    echo "Fold $((fold+1)) completed!"
    echo ""
done

echo "========================================"
echo "5-Fold CV Completed!"
echo "========================================"

# Python으로 결과 집계
python << 'EOF'
import glob
import re
import numpy as np

results = []
for fold in range(5):
    log_file = f"logs/improved_5fold/fold_{fold}/log.txt"
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            # Final test 결과 추출
            match = re.search(r'All views: Loss ([\d.]+), Acc ([\d.]+)%', content)
            if match:
                loss = float(match.group(1))
                acc = float(match.group(2))
                results.append((acc, loss))
                print(f"Fold {fold+1}: Test Acc = {acc:.2f}%, Loss = {loss:.4f}")
    except Exception as e:
        print(f"Fold {fold+1}: Error reading results - {e}")

if results:
    accs, losses = zip(*results)
    print(f"\nAverage: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
    print(f"Loss: {np.mean(losses):.4f}")
else:
    print("\nNo results found!")
EOF
