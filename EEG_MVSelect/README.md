# EEG Multi-View Selection (EEG-MVSelect)

## Overview
이 프로젝트는 **뇌파(EEG) 연결성 이미지의 효율적인 멀티뷰 학습**을 위한 뷰 선택(View Selection) 기법을 제안합니다. 
같은 피험자의 뇌파 데이터라도 연결성 측정 방법(FCM, PCC, PLV), 주파수 밴드(Delta, Theta, Alpha, Beta, Gamma), 
채널 조합에 따라 다양한 연결성 이미지가 생성됩니다. 

본 연구는 MVSelect 아키텍처를 활용하여 **높은 연산 비용 문제를 해결**하면서도 
**가장 유용한 연결성 이미지만을 자동으로 선택**하여 분류 성능을 유지하는 것을 목표로 합니다.

## 주요 특징
- 🧠 **다중 연결성 표현**: FCM (Functional Connectivity Matrix), PCC (Pearson Correlation), PLV (Phase Locking Value)
- 📊 **주파수 밴드 분석**: Delta, Theta, Alpha, Beta, Gamma 밴드별 연결성
- 🎯 **효율적인 뷰 선택**: 모든 이미지를 사용하지 않고 가장 유용한 이미지만 선택
- ⚡ **연산 비용 절감**: 멀티뷰 시스템의 높은 계산 비용 문제 해결
- 🔄 **적응적 학습**: 강화학습 기반의 동적 뷰 선택 전략

## 데이터 구조
```
FCM_Images_HERMES_v2/
├── FCM/          # Functional Connectivity Matrix (PCC + PLV 혼합)
├── PCC/          # Pearson Correlation Coefficient
└── PLV/          # Phase Locking Value
```

각 폴더 내부는 다음과 같은 구조를 가집니다:
```
{Method}/
├── {Subject_ID}/
│   ├── {Band}_connectivity.png
│   └── ...
└── ...
```

## 설치 방법

### 환경 요구사항
- Python >= 3.7
- PyTorch >= 1.8
- CUDA (GPU 학습 권장)

### 의존성 설치
```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 데이터 준비
뇌파 연결성 이미지가 다음 경로에 위치해야 합니다:
```
/home/work/skku/icip/FCM_Images_HERMES_v2/
```

### 2. 기본 학습
```bash
# Step 1: Task Network 학습 (분류기)
python main.py --dataset eeg_connectivity --epochs 50

# Step 2: MVSelect 모듈 학습
python main.py --dataset eeg_connectivity --steps 2 --epochs 30
```

### 3. 조인트 학습
```bash
# Task Network와 MVSelect 동시 학습
python main.py --dataset eeg_connectivity --steps 2 --joint_training
```

### 4. 평가
```bash
# 학습된 모델 평가
python main.py --dataset eeg_connectivity --eval --resume MODEL_PATH
```

## 모델 아키텍처

### 전체 구조
```
Input: Multiple Connectivity Images (FCM, PCC, PLV × Bands)
    ↓
Feature Extractor (ResNet18/VGG11)
    ↓
View Selection Module (MVSelect)
    ↓
Aggregation (Max/Mean Pooling)
    ↓
Classification Head
    ↓
Output: Subject/Task Classification
```

### View Selection 전략
- **초기화**: 랜덤 또는 특정 연결성 방법으로 시작
- **순차적 선택**: 강화학습을 통해 가장 유용한 뷰를 단계별로 추가
- **보상 설계**: 분류 정확도 향상을 기반으로 한 보상 함수

## 실험 설정

### 하이퍼파라미터
- Learning Rate: 5e-5 (Task Network), 1e-4 (MVSelect)
- Batch Size: 8
- Optimizer: Adam
- Aggregation: Max Pooling
- Steps: 2-4 (선택할 뷰의 개수)

## 결과 분석

학습이 완료되면 다음 정보가 `logs/` 폴더에 저장됩니다:
- 학습 로그 및 성능 메트릭
- 선택된 뷰의 통계
- 연산 비용 분석

## 프로젝트 구조
```
EEG_MVSelect/
├── src/
│   ├── models/
│   │   ├── eeg_mvselect.py      # EEG용 MVSelect 모델
│   │   ├── mvselect.py          # 뷰 선택 모듈
│   │   ├── resnet.py            # ResNet 백본
│   │   └── multiview_base.py    # 기본 멀티뷰 클래스
│   ├── datasets/
│   │   └── eeg_connectivity.py  # EEG 연결성 데이터셋
│   ├── utils/
│   │   └── ...                  # 유틸리티 함수들
│   └── loss/
│       └── ...                  # 손실 함수들
├── main.py                      # 메인 학습 스크립트
├── requirements.txt
└── README.md
```

## 참고 문헌
- Original MVSelect: "Learning to Select Camera Views: Efficient Multiview Understanding at Few Glances"
- EEG Connectivity Analysis: Various functional connectivity methods

## License
MIT License

## Contact
For questions and support, please contact the project maintainer.
