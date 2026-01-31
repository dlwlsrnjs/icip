"""
EEG Multi-View Selection Model

뇌파 연결성 이미지를 위한 멀티뷰 선택 모델
FCM, PCC, PLV 등 다양한 연결성 측정 방법과 주파수 밴드로 구성된
멀티뷰 데이터에서 가장 유용한 뷰를 자동으로 선택
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .multiview_base import MultiviewBase
from .mvselect import CamSelect


class EEGMVSelect(MultiviewBase):
    """
    EEG 연결성 이미지를 위한 Multi-View Selection 모델
    
    Args:
        dataset: EEGConnectivityDataset 인스턴스
        arch: 백본 아키텍처 ('resnet18', 'resnet34', 'vgg11')
        aggregation: 뷰 통합 방법 ('max', 'mean')
        pretrained: 사전학습 가중치 사용 여부
    """
    
    def __init__(self, dataset, arch='resnet18', aggregation='max', pretrained=True):
        super().__init__(dataset, aggregation)
        
        # 백본 네트워크 설정
        if arch == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            self.base = nn.Sequential(*list(base_model.children())[:-2])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            base_dim = 512
            
        elif arch == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            self.base = nn.Sequential(*list(base_model.children())[:-2])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            base_dim = 512
            
        elif arch == 'vgg11':
            base_model = models.vgg11(pretrained=pretrained)
            self.base = base_model.features
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            base_dim = 512
            
        elif arch == 'vgg16':
            base_model = models.vgg16(pretrained=pretrained)
            self.base = base_model.features
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            base_dim = 512
            
        else:
            raise ValueError(f"Unsupported architecture: {arch}. "
                           f"Choose from ['resnet18', 'resnet34', 'vgg11', 'vgg16']")
        
        self.arch = arch
        self.base_dim = base_dim
        
        # 분류기 (피험자/상태 분류)
        if 'vgg' in arch:
            self.classifier = nn.Sequential(
                nn.Linear(base_dim * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, dataset.num_class)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(base_dim, dataset.num_class)
            )
        
        # View Selection 모듈
        # kernel_size=1: feature map이 작을 때 사용
        self.select_module = CamSelect(
            num_cam=dataset.num_cam,
            hidden_dim=base_dim,
            kernel_size=1,
            aggregation=aggregation
        )
        
        print(f"EEG-MVSelect Model Initialized:")
        print(f"  Architecture: {arch}")
        print(f"  Base dimension: {base_dim}")
        print(f"  Number of views: {dataset.num_cam}")
        print(f"  Number of classes: {dataset.num_class}")
        print(f"  Aggregation: {aggregation}")
        print(f"  Pretrained: {pretrained}")
    
    def get_feat(self, imgs, M=None, down=1, visualize=False):
        """
        이미지에서 특징 추출
        
        Args:
            imgs: [B, N, C, H, W] 형태의 입력 이미지
            M: 사용하지 않음 (호환성 유지)
            down: 다운샘플링 비율
            visualize: 시각화 플래그
            
        Returns:
            feat: [B, N, C, H, W] 형태의 특징맵
            aux_res: None (보조 출력 없음)
        """
        B, N, C, H, W = imgs.shape
        
        # 다운샘플링 (필요시)
        if down > 1:
            imgs = F.interpolate(imgs.flatten(0, 1), scale_factor=1/down, mode='bilinear', align_corners=False)
        else:
            imgs = imgs.flatten(0, 1)  # [B*N, C, H, W]
        
        # 백본 네트워크를 통한 특징 추출
        imgs_feat = self.base(imgs)
        imgs_feat = self.avgpool(imgs_feat)
        
        # [B*N, C, H, W] -> [B, N, C, H, W]
        _, C_feat, H_feat, W_feat = imgs_feat.shape
        imgs_feat = imgs_feat.view(B, N, C_feat, H_feat, W_feat)
        
        return imgs_feat, None
    
    def get_output(self, overall_feat, visualize=False):
        """
        통합된 특징에서 최종 출력 생성
        
        Args:
            overall_feat: [B, C, H, W] 형태의 통합 특징
            visualize: 시각화 플래그
            
        Returns:
            output: [B, num_class] 형태의 분류 결과
        """
        # Flatten
        overall_feat = torch.flatten(overall_feat, 1)
        
        # 분류
        output = self.classifier(overall_feat)
        
        return output
    
    def forward_with_selection_info(self, imgs, M=None, down=1, init_prob=None, 
                                    steps=0, keep_cams=None, visualize=False):
        """
        선택 정보와 함께 forward pass 수행
        
        Returns:
            output: 분류 결과
            aux_res: 보조 출력
            selection_res: 선택 정보 (log_probs, values, actions, entropies)
            selected_views: 선택된 뷰 인덱스 리스트
        """
        # 일반 forward
        output, aux_res, selection_res = self.forward(
            imgs, M, down, init_prob, steps, keep_cams, visualize
        )
        
        # 선택된 뷰 추출
        selected_views = []
        if selection_res[2] is not None:  # actions
            for action in selection_res[2]:
                selected_views.append(action.argmax(dim=-1).cpu().numpy())
        
        return output, aux_res, selection_res, selected_views


class EEGMVSelectWithAuxTask(EEGMVSelect):
    """
    보조 태스크가 있는 EEG-MVSelect
    예: 주 태스크(피험자 분류) + 보조 태스크(상태 분류)
    """
    
    def __init__(self, dataset, aux_num_class, arch='resnet18', 
                 aggregation='max', pretrained=True):
        super().__init__(dataset, arch, aggregation, pretrained)
        
        # 보조 태스크 분류기
        if 'vgg' in arch:
            self.aux_classifier = nn.Sequential(
                nn.Linear(self.base_dim * 7 * 7, 2048),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(2048, aux_num_class)
            )
        else:
            self.aux_classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.base_dim, aux_num_class)
            )
        
        print(f"  Auxiliary task classes: {aux_num_class}")
    
    def get_output(self, overall_feat, visualize=False):
        """
        통합된 특징에서 주 태스크와 보조 태스크 출력 생성
        
        Returns:
            (main_output, aux_output): 주 태스크 및 보조 태스크 결과
        """
        overall_feat_flat = torch.flatten(overall_feat, 1)
        
        main_output = self.classifier(overall_feat_flat)
        aux_output = self.aux_classifier(overall_feat_flat)
        
        return main_output, aux_output


def create_eeg_mvselect_model(dataset, config):
    """
    설정에 따라 EEG-MVSelect 모델 생성
    
    Args:
        dataset: EEGConnectivityDataset 인스턴스
        config: 모델 설정 딕셔너리
            - arch: 아키텍처
            - aggregation: 통합 방법
            - pretrained: 사전학습 여부
            - aux_task: 보조 태스크 사용 여부
            - aux_num_class: 보조 태스크 클래스 수
    
    Returns:
        model: EEG-MVSelect 모델 인스턴스
    """
    if config.get('aux_task', False):
        model = EEGMVSelectWithAuxTask(
            dataset=dataset,
            aux_num_class=config.get('aux_num_class', 2),
            arch=config.get('arch', 'resnet18'),
            aggregation=config.get('aggregation', 'max'),
            pretrained=config.get('pretrained', True)
        )
    else:
        model = EEGMVSelect(
            dataset=dataset,
            arch=config.get('arch', 'resnet18'),
            aggregation=config.get('aggregation', 'max'),
            pretrained=config.get('pretrained', True)
        )
    
    return model


if __name__ == '__main__':
    """모델 테스트"""
    from src.datasets import EEGConnectivityDataset
    
    # 데이터셋 생성
    dataset = EEGConnectivityDataset(
        root_dir='/home/work/skku/icip/FCM_Images_HERMES_v2',
        split='train'
    )
    
    # 모델 생성
    model = EEGMVSelect(dataset, arch='resnet18', aggregation='max')
    model = model.cuda()
    model.eval()
    
    print("\nModel Test:")
    print("="*50)
    
    # 더미 입력
    B, N, C, H, W = 2, dataset.num_cam, 3, 224, 224
    imgs = torch.randn(B, N, C, H, W).cuda()
    keep_cams = torch.ones(B, N, dtype=torch.bool).cuda()
    
    # Forward pass (no selection)
    with torch.no_grad():
        output, _, _ = model(imgs, down=1)
        print(f"Output shape (no selection): {output.shape}")
    
    # Forward pass (with selection)
    init_prob = F.one_hot(torch.tensor([0, 0]), num_classes=N).cuda()
    with torch.no_grad():
        output, _, selection_res = model(imgs, init_prob=init_prob, steps=2, keep_cams=keep_cams)
        print(f"Output shape (with selection): {output.shape}")
        print(f"Number of selection steps: {len(selection_res[2])}")
    
    print("="*50)
    print("Model test completed successfully!")
