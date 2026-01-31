import os
import glob
import random
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np


class EEGConnectivityDataset(Dataset):
    """
    뇌파 연결성 이미지 데이터셋
    
    FCM, PCC, PLV 세 가지 연결성 측정 방법과
    여러 주파수 밴드(Delta, Theta, Alpha, Beta, Gamma)의 조합으로 구성된
    멀티뷰 이미지 데이터셋
    """
    
    # 연결성 측정 방법
    CONNECTIVITY_METHODS = ['FCM', 'PCC', 'PLV']
    
    # 주파수 밴드
    FREQUENCY_BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    def __init__(self, root_dir, split='train', transform=None, 
                 connectivity_methods=None, frequency_bands=None,
                 image_size=224, train_ratio=0.7, val_ratio=0.1,
                 label_file='/home/work/skku/icip/subject_labels.csv',
                 use_cv=False, cv_fold=0, n_folds=5):
        """
        Args:
            root_dir: 데이터 루트 디렉토리 (FCM_Images_HERMES_v2)
            split: 'train', 'val', 'test' 중 하나 (use_cv=False일 때)
            transform: 이미지 변환
            use_cv: Cross-Validation 사용 여부
            cv_fold: 현재 fold 번호 (0~n_folds-1)
            n_folds: 총 fold 개수
            connectivity_methods: 사용할 연결성 방법 리스트 (None이면 모두 사용)
            frequency_bands: 사용할 주파수 밴드 리스트 (None이면 모두 사용)
            image_size: 이미지 크기
            train_ratio: 학습 데이터 비율
            val_ratio: 검증 데이터 비율
            label_file: 피험자 레이블 CSV 파일 경로 (ADHD/Control)
        """
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.use_cv = use_cv
        self.cv_fold = cv_fold
        self.n_folds = n_folds
        
        # 사용할 연결성 방법과 주파수 밴드 설정
        self.connectivity_methods = connectivity_methods if connectivity_methods else self.CONNECTIVITY_METHODS
        self.frequency_bands = frequency_bands if frequency_bands else self.FREQUENCY_BANDS
        
        # 총 뷰 개수 계산
        self.num_views = len(self.connectivity_methods) * len(self.frequency_bands)
        self.num_cam = self.num_views  # MVSelect 호환성을 위해
        
        # 레이블 파일 로드 (ADHD vs Control)
        self.subject_to_group = {}
        with open(label_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.subject_to_group[row['Subject_ID']] = row['Group']
        
        # 클래스 정보 (ADHD=0, Control=1)
        self.classes = ['ADHD', 'Control']
        self.num_class = len(self.classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # 데이터 로드
        self.samples = self._load_dataset(train_ratio, val_ratio, use_cv, cv_fold, n_folds)
        
        # 기본 변환 설정
        if transform is None:
            if split == 'train':
                # Connectivity 이미지에 적합한 최소 Augmentation
                # 공간 구조가 중요하므로 flip/rotation 제거
                self.transform = T.Compose([
                    T.Resize((image_size, image_size)),
                    T.ColorJitter(brightness=0.1, contrast=0.1),  # 약한 색상 변화만
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = T.Compose([
                    T.Resize((image_size, image_size)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
            
        # 클래스별 샘플 수 계산
        adhd_count = sum(1 for s in self.samples if self.subject_to_group.get(s['subject_id']) == 'ADHD')
        control_count = sum(1 for s in self.samples if self.subject_to_group.get(s['subject_id']) == 'Control')
        
        print(f"EEG Connectivity Dataset - {split} split:")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Classes: {self.num_class} (ADHD: {adhd_count}, Control: {control_count})")
        print(f"  Number of views per sample: {self.num_views}")
        print(f"  Connectivity methods: {self.connectivity_methods}")
        print(f"  Frequency bands: {self.frequency_bands}")
    
    def _load_dataset(self, train_ratio, val_ratio, use_cv=False, cv_fold=0, n_folds=5):
        """데이터셋 로드 및 train/val/test 분할 또는 CV fold 분할"""
        
        # 먼저 모든 파일에서 피험자 ID 추출
        # 파일명 형식: v107_alpha.png, v108_beta.png 등
        all_subjects = set()
        
        for method in self.connectivity_methods:
            method_dir = os.path.join(self.root_dir, method)
            if not os.path.exists(method_dir):
                print(f"Warning: {method_dir} does not exist. Skipping.")
                continue
            
            # 모든 이미지 파일 찾기
            image_files = glob.glob(os.path.join(method_dir, "*.png")) + \
                         glob.glob(os.path.join(method_dir, "*.jpg"))
            
            # 피험자 ID 추출 (파일명에서 '_' 앞부분)
            for img_path in image_files:
                filename = os.path.basename(img_path)
                # v107_alpha.png -> v107
                subject_id = filename.split('_')[0]
                all_subjects.add(subject_id)
        
        # 피험자별 샘플 생성
        samples = []
        for subject_id in sorted(all_subjects):
            sample = {
                'subject_id': subject_id,
                'paths': {}
            }
            
            # 각 method와 band 조합에 대한 파일 찾기
            all_found = True
            for method in self.connectivity_methods:
                method_dir = os.path.join(self.root_dir, method)
                if not os.path.exists(method_dir):
                    all_found = False
                    break
                
                for band in self.frequency_bands:
                    # 파일명 패턴: {subject_id}_{band}.png
                    patterns = [
                        f"{subject_id}_{band}.png",
                        f"{subject_id}_{band}.jpg",
                        f"{subject_id}_{band.upper()}.png",
                        f"{subject_id}_{band.capitalize()}.png",
                    ]
                    
                    img_path = None
                    for pattern in patterns:
                        full_path = os.path.join(method_dir, pattern)
                        if os.path.exists(full_path):
                            img_path = full_path
                            break
                    
                    if img_path:
                        view_key = f"{method}_{band}"
                        sample['paths'][view_key] = img_path
                    else:
                        all_found = False
                        break
                
                if not all_found:
                    break
            
            # 모든 뷰가 있는 샘플만 추가
            if all_found and len(sample['paths']) == self.num_views:
                samples.append(sample)
        
        complete_samples = samples
        
        # 피험자별로 그룹화
        subject_samples = {}
        for sample in complete_samples:
            subj = sample['subject_id']
            if subj not in subject_samples:
                subject_samples[subj] = []
            subject_samples[subj].append(sample)
        
        # 클래스별로 피험자 분리
        subjects = sorted(list(subject_samples.keys()))
        adhd_subjects = [s for s in subjects if self.subject_to_group.get(s) == 'ADHD']
        control_subjects = [s for s in subjects if self.subject_to_group.get(s) == 'Control']
        
        random.seed(42)
        random.shuffle(adhd_subjects)
        random.shuffle(control_subjects)
        
        if use_cv:
            # Cross-Validation: 각 클래스를 n_folds로 나누기
            # 각 fold는 test로 사용, 나머지는 train으로 사용
            n_adhd = len(adhd_subjects)
            n_control = len(control_subjects)
            
            # Fold 크기 계산
            fold_size_adhd = n_adhd // n_folds
            fold_size_control = n_control // n_folds
            
            # 현재 fold의 test subjects
            adhd_start = cv_fold * fold_size_adhd
            adhd_end = (cv_fold + 1) * fold_size_adhd if cv_fold < n_folds - 1 else n_adhd
            control_start = cv_fold * fold_size_control
            control_end = (cv_fold + 1) * fold_size_control if cv_fold < n_folds - 1 else n_control
            
            adhd_test = adhd_subjects[adhd_start:adhd_end]
            control_test = control_subjects[control_start:control_end]
            
            # 나머지는 train
            adhd_train = adhd_subjects[:adhd_start] + adhd_subjects[adhd_end:]
            control_train = control_subjects[:control_start] + control_subjects[control_end:]
            
            if self.split == 'train':
                selected_subjects = adhd_train + control_train
            elif self.split == 'test':
                selected_subjects = adhd_test + control_test
            else:  # val - train의 일부를 val로 사용
                # Train의 마지막 10%를 val로 사용
                n_val_adhd = max(1, len(adhd_train) // 10)
                n_val_control = max(1, len(control_train) // 10)
                selected_subjects = adhd_train[-n_val_adhd:] + control_train[-n_val_control:]
        else:
            # 기존 Hold-out 방식: train/val/test 분할 (Stratified)
            n_adhd = len(adhd_subjects)
            n_adhd_train = int(n_adhd * train_ratio)
            n_adhd_val = int(n_adhd * val_ratio)
            
            n_control = len(control_subjects)
            n_control_train = int(n_control * train_ratio)
            n_control_val = int(n_control * val_ratio)
            
            # ADHD split
            adhd_train = adhd_subjects[:n_adhd_train]
            adhd_val = adhd_subjects[n_adhd_train:n_adhd_train + n_adhd_val]
            adhd_test = adhd_subjects[n_adhd_train + n_adhd_val:]
            
            # Control split
            control_train = control_subjects[:n_control_train]
            control_val = control_subjects[n_control_train:n_control_train + n_control_val]
            control_test = control_subjects[n_control_train + n_control_val:]
            
            # 합치기
            train_subjects = adhd_train + control_train
            val_subjects = adhd_val + control_val
            test_subjects = adhd_test + control_test
            
            if self.split == 'train':
                selected_subjects = train_subjects
            elif self.split == 'val':
                selected_subjects = val_subjects
            else:  # test
                selected_subjects = test_subjects
        
        # 선택된 피험자의 샘플만 반환
        result = []
        for subj in selected_subjects:
            result.extend(subject_samples[subj])
        
        return result
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            images: [num_views, C, H, W] 형태의 텐서
            label: 클래스 인덱스 (0: ADHD, 1: Control)
            keep_cams: [num_views] 사용 가능한 뷰 마스크 (모두 True)
        """
        sample = self.samples[idx]
        subject_id = sample['subject_id']
        
        # ADHD/Control 레이블
        group = self.subject_to_group.get(subject_id, 'Control')  # 기본값은 Control
        label = self.class_to_idx[group]
        
        # 모든 뷰의 이미지 로드
        images = []
        view_order = []
        
        for method in self.connectivity_methods:
            for band in self.frequency_bands:
                view_key = f"{method}_{band}"
                if view_key in sample['paths']:
                    img_path = sample['paths'][view_key]
                    try:
                        img = Image.open(img_path).convert('RGB')
                        if self.transform:
                            img = self.transform(img)
                        images.append(img)
                        view_order.append(view_key)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        # 에러 시 검은 이미지로 대체
                        img = torch.zeros(3, self.image_size, self.image_size)
                        images.append(img)
                        view_order.append(view_key)
        
        # [N, C, H, W] 형태로 스택
        images = torch.stack(images)
        
        # 모든 카메라/뷰 사용 가능
        keep_cams = torch.ones(self.num_views, dtype=torch.bool)
        
        return images, label, keep_cams
    
    def get_view_name(self, view_idx):
        """뷰 인덱스에 해당하는 이름 반환"""
        method_idx = view_idx // len(self.frequency_bands)
        band_idx = view_idx % len(self.frequency_bands)
        return f"{self.connectivity_methods[method_idx]}_{self.frequency_bands[band_idx]}"


def test_dataset():
    """데이터셋 테스트 함수"""
    root = '/home/work/skku/icip/FCM_Images_HERMES_v2'
    
    # 데이터셋 생성
    train_dataset = EEGConnectivityDataset(root, split='train')
    val_dataset = EEGConnectivityDataset(root, split='val')
    test_dataset = EEGConnectivityDataset(root, split='test')
    
    print("\n" + "="*50)
    print("Dataset Test Results:")
    print("="*50)
    
    # 샘플 로드 테스트
    if len(train_dataset) > 0:
        images, label, keep_cams = train_dataset[0]
        print(f"\nSample batch shape:")
        print(f"  Images: {images.shape}")
        print(f"  Label: {label}")
        print(f"  Keep_cams: {keep_cams.shape}")
        
        print(f"\nView names:")
        for i in range(train_dataset.num_views):
            print(f"  View {i}: {train_dataset.get_view_name(i)}")
    else:
        print("\nWarning: No samples found in training set!")


if __name__ == '__main__':
    test_dataset()
