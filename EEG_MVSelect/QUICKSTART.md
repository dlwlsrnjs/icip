# Quick Start Guide

## 1. 환경 설정

```bash
cd /home/work/skku/icip/EEG_MVSelect

# 의존성 설치
pip install -r requirements.txt
```

## 2. 데이터 확인

먼저 데이터가 올바르게 구성되어 있는지 확인:

```bash
python check_data.py
```

예상 출력:
- FCM, PCC, PLV 각 폴더의 피험자 수
- 각 피험자별 주파수 밴드 이미지 개수
- 완전한 데이터를 가진 피험자 수

## 3. 빠른 테스트

모든 것이 정상적으로 작동하는지 확인:

```bash
python quick_test.py
```

이 스크립트는 다음을 테스트합니다:
1. 데이터셋 로딩
2. 샘플 로드
3. 모델 생성
4. Forward pass

## 4. 학습

### Option A: 쉘 스크립트 사용 (권장)

```bash
# 전체 학습 파이프라인 실행
./train.sh
```

이 스크립트는:
1. Step 1: 모든 뷰를 사용한 기본 분류기 학습
2. Step 2: MVSelect 모듈 학습

### Option B: 수동 실행

**Step 1: 기본 분류기 학습**
```bash
python main.py \
    --arch resnet18 \
    --aggregation max \
    --epochs 50 \
    --batch_size 8 \
    --lr 5e-5 \
    --steps 0 \
    --scheduler cosine
```

**Step 2: MVSelect 학습**
```bash
python main.py \
    --arch resnet18 \
    --aggregation max \
    --epochs 30 \
    --batch_size 8 \
    --lr 5e-5 \
    --select_lr 1e-4 \
    --base_lr_ratio 0.1 \
    --steps 2 \
    --init_cam random \
    --rl_loss_weight 0.1 \
    --scheduler cosine
```

## 5. 평가

학습된 모델 평가:

```bash
./evaluate.sh logs/YOUR_EXPERIMENT_DIR/best_model.pth
```

또는:

```bash
python main.py \
    --eval \
    --resume logs/YOUR_EXPERIMENT_DIR/best_model.pth \
    --steps 2
```

## 6. 주요 하이퍼파라미터

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--arch` | 백본 아키텍처 | `resnet18` |
| `--aggregation` | 뷰 통합 방법 | `max` |
| `--steps` | 선택할 뷰 개수 (0=전체) | `0` |
| `--epochs` | 학습 에폭 | `50` |
| `--batch_size` | 배치 크기 | `8` |
| `--lr` | 기본 학습률 | `5e-5` |
| `--select_lr` | MVSelect 학습률 | `1e-4` |
| `--base_lr_ratio` | Base 네트워크 학습률 비율 | `1.0` |
| `--rl_loss_weight` | 강화학습 손실 가중치 | `0.1` |

## 7. 학습 결과 확인

학습 로그는 `logs/` 디렉토리에 저장됩니다:

```
logs/
└── resnet18_max_steps2_lr5e-05_bs8_TIMESTAMP/
    ├── log.txt              # 학습 로그
    ├── best_model.pth       # 최고 성능 모델
    └── checkpoint_*.pth     # 에폭별 체크포인트
```

## 8. 시각화

뷰 선택 통계 분석:

```python
from src.utils import analyze_view_selection_statistics
from src.models import EEGMVSelect
from src.datasets import EEGConnectivityDataset
from torch.utils.data import DataLoader

# 데이터 및 모델 로드
dataset = EEGConnectivityDataset(...)
model = EEGMVSelect(...)
# 체크포인트 로드
checkpoint = torch.load('path/to/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 통계 분석
dataloader = DataLoader(dataset, batch_size=8)
analyze_view_selection_statistics(model, dataloader, steps=2, save_dir='results')
```

## 문제 해결

### GPU 메모리 부족
- `--batch_size`를 줄이기 (예: 4 또는 2)
- `--down 2`로 이미지 다운샘플링

### 데이터 로드 에러
- `check_data.py`로 데이터 구조 확인
- 이미지 파일 경로 및 파일명 패턴 확인

### 학습이 너무 느림
- `--num_workers` 증가 (예: 8)
- SSD 사용 권장

## 추가 정보

자세한 내용은 메인 [README.md](README.md) 참고
