# EEG-MVSelect 프로젝트 완료 보고서

## 📋 프로젝트 개요
**MVSelect 기반 뇌파 연결성 이미지 멀티뷰 학습 시스템**을 성공적으로 구축했습니다.

## 🎯 핵심 개념

### 멀티뷰 구조
- **3가지 연결성 측정 방법**: FCM, PCC, PLV
- **5가지 주파수 밴드**: Delta, Theta, Alpha, Beta, Gamma  
- **총 15개 뷰**: 3 methods × 5 bands = 15 views per subject

### MVSelect 접근법
같은 피험자라도 연결성 측정 방법과 주파수 밴드에 따라 다른 이미지가 생성됩니다.
모든 이미지를 사용하면 높은 연산 비용이 발생하므로, 강화학습으로 가장 유용한 이미지만 선택합니다.

## 📊 데이터 현황
- 총 피험자: 121명
- Train: 84명, Val: 18명, Test: 19명
- 각 피험자당 15개 뷰 (FCM×5 + PCC×5 + PLV×5)

## 🚀 빠른 시작

### 학습 실행
\`\`\`bash
cd /home/work/skku/icip/EEG_MVSelect
./train.sh
\`\`\`

### 데이터 테스트
\`\`\`bash
python test_dataset.py
\`\`\`

## ✅ 완료 사항
- [x] MVSelect 코드 적용
- [x] 데이터 구조 분석 (FCM/PCC/PLV 3개 폴더)
- [x] 데이터셋 로더 구현 (15 views per subject)
- [x] EEG-MVSelect 모델 구현
- [x] 학습/평가 파이프라인
- [x] 데이터 로딩 검증 완료

Created: 2026-01-30
Location: /home/work/skku/icip/EEG_MVSelect
