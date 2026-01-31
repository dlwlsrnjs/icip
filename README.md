# EEG Connectivity Multi-View Learning for ADHD Classification

EEG connectivity 이미지를 이용한 ADHD vs Control 분류를 위한 Multi-View Learning 프로젝트

## 📌 프로젝트 개요

- **목표**: EEG connectivity 이미지를 활용하여 ADHD 환자와 정상 대조군을 분류
- **데이터**: 121명 (ADHD: 61명, Control: 60명)
- **Multi-View**: 3가지 connectivity 방법 (FCM, PCC, PLV) × 5개 주파수 밴드 = 15 views
- **모델**: MVSelect + Paper-Inspired Att-CNN (Neuroinformatics 2024)

## 🏗️ 프로젝트 구조

```
icip/
├── EEG_MVSelect/              # Main project
│   ├── src/
│   │   ├── models/
│   │   │   ├── paper_exact_attention.py    # Lightweight Att-CNN (360K params)
│   │   │   ├── resnet.py                   # ResNet variants
│   │   │   ├── mvselect.py                 # Multi-view selection (DQN)
│   │   │   └── multiview_base.py
│   │   ├── datasets/
│   │   │   └── eeg_connectivity.py         # EEG dataset loader with CV
│   │   └── trainer.py                      # Training & evaluation
│   ├── main.py                             # Entry point
│   ├── train_improved.sh                   # Optimized training script
│   ├── train_5fold_cv.sh                   # 5-Fold Cross-Validation
│   └── requirements.txt
├── FCM_Images_HERMES_v2/      # Dataset (not included)
└── subject_labels.csv         # Subject labels
```

## 🎯 주요 기능

### 1. **Paper-Inspired Att-CNN**
Neuroinformatics 2024 논문 기반 경량 모델:
- 3 Conv layers (32→64→128 filters)
- Channel Attention (reduction=8)
- 360K parameters (vs ResNet50 25M)
- BatchNorm + Dropout regularization

### 2. **Multi-View Learning**
- 15개 view를 통합하여 robust한 분류
- Max aggregation across views
- Optional: MVSelect로 중요한 view만 선택 (3/15)

### 3. **5-Fold Cross-Validation**
- 신뢰성 있는 성능 평가
- 각 fold마다 stratified split (class balance 유지)
- Train: ~97, Val: 8, Test: ~24 samples per fold

## 🚀 빠른 시작

### 설치
```bash
cd EEG_MVSelect
pip install -r requirements.txt
```

### 데이터셋 준비
```
FCM_Images_HERMES_v2/
├── FCM/
│   └── v{subject_id}_{band}.png
├── PCC/
│   └── v{subject_id}_{band}.png
└── PLV/
    └── v{subject_id}_{band}.png
```

### 학습
```bash
# 개선된 설정으로 5-Fold CV
./train_improved.sh

# 단일 학습
python main.py \
    --use_paper_exact \
    --data_root ../FCM_Images_HERMES_v2 \
    --batch_size 16 \
    --epochs 80 \
    --lr 5e-4 \
    --weight_decay 1e-4
```

## 📊 성능 결과

### 개선 전 (과도한 augmentation + 높은 regularization)
- 5-Fold CV Average: **61.67% ± 7.17%**
- Train Acc: 60-70% (underfitting)
- Val Acc: 50-62.5% (random 수준)

### 개선 후 (최소 augmentation + 낮은 regularization)
- 현재 진행 중
- Fold 2 Test: **75.00%**
- Fold 3 Val: **87.50%**
- Train Acc: 80-88% (적절한 학습)

### 주요 개선사항
1. **Data Augmentation 최소화**: Flip/Rotation 제거 → ColorJitter만 사용
2. **Regularization 완화**: Dropout 0.5→0.3, Weight Decay 5e-4→1e-4
3. **Learning Rate 감소**: 1e-3 → 5e-4

## 🔧 하이퍼파라미터

```python
# Model
architecture = "Paper-Inspired Att-CNN"
num_params = 360322
input_size = (224, 224)
num_views = 15

# Training
batch_size = 16
epochs = 80
lr = 5e-4
weight_decay = 1e-4
optimizer = "SGD with momentum 0.9"
scheduler = "Cosine Annealing"

# Augmentation (Train only)
ColorJitter(brightness=0.1, contrast=0.1)
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

## 📈 모델 아키텍처

```
Input: [B, 15, 3, 224, 224]
  ↓
Conv1(32) + BN + ReLU + MaxPool(2)  → [B, 15, 32, 112, 112]
  ↓
Conv2(64) + BN + ReLU               → [B, 15, 64, 112, 112]
  ↓
Conv3(128) + BN + ReLU              → [B, 15, 128, 112, 112]
  ↓
Channel Attention (reduction=8)     → [B, 15, 128, 112, 112]
  ↓
MaxPool(2) + AdaptiveAvgPool(4×4)   → [B, 15, 128, 4, 4]
  ↓
Max Aggregation across views        → [B, 128, 4, 4]
  ↓
Flatten                             → [B, 2048]
  ↓
Dense(128) + ReLU + Dropout(0.2)    → [B, 128]
  ↓
Dense(2)                            → [B, 2]
```

## 📝 참고 문헌

- Neuroinformatics 2024: EEG Connectivity with Channel Attention for ADHD Classification
- MVSelect: DQN-based Multi-View Selection

## 👤 작성자

dlwlsrnjs

## 📄 라이선스

MIT License
