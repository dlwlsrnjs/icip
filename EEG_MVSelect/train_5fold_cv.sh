#!/bin/bash

# 5-Fold Cross-Validation for Paper-Inspired Att-CNN
# Stage 1 only (all views)

LOG_DIR="logs/paper_exact_5fold"
mkdir -p ${LOG_DIR}

echo "========================================"
echo "Paper-Inspired Att-CNN 5-Fold CV"
echo "========================================"
echo ""

# 5개 fold 학습
for fold in {0..4}
do
    echo "========================================" echo "Fold $((fold + 1))/5"
    echo "========================================"
    
    python main.py \
        --use_paper_exact \
        --data_root /home/work/skku/icip/FCM_Images_HERMES_v2 \
        --image_size 224 \
        --batch_size 16 \
        --epochs 80 \
        --lr 1e-3 \
        --weight_decay 5e-4 \
        --steps 0 \
        --save_dir ${LOG_DIR}/fold_${fold} \
        --seed 42 \
        --use_cv \
        --cv_fold ${fold} \
        --n_folds 5
    
    echo "Fold $((fold + 1)) completed!"
    echo ""
done

echo "========================================"
echo "5-Fold CV Completed!"
echo "========================================"

# 평균 성능 계산
python << 'EOF'
import re
import glob

folds = []
for fold in range(5):
    log_files = glob.glob(f'logs/paper_exact_5fold/fold_{fold}/*/log.txt')
    if log_files:
        with open(log_files[0], 'r') as f:
            content = f.read()
            # Test accuracy 추출
            matches = re.findall(r'All views: Loss ([\d\.]+), Acc ([\d\.]+)%', content)
            if matches:
                test_loss, test_acc = matches[-1]  # 마지막이 test
                folds.append({
                    'fold': fold,
                    'test_acc': float(test_acc),
                    'test_loss': float(test_loss)
                })

if folds:
    print("\n=== 5-Fold CV Results ===")
    for f in folds:
        print(f"Fold {f['fold']+1}: Test Acc = {f['test_acc']:.2f}%, Loss = {f['test_loss']:.4f}")
    
    avg_acc = sum(f['test_acc'] for f in folds) / len(folds)
    avg_loss = sum(f['test_loss'] for f in folds) / len(folds)
    std_acc = (sum((f['test_acc'] - avg_acc)**2 for f in folds) / len(folds))**0.5
    
    print(f"\nAverage: {avg_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Loss: {avg_loss:.4f}")
else:
    print("No results found!")
EOF
