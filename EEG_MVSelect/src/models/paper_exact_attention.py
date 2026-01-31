"""
Neuroinformatics 2024 논문 정확히 따라한 Att-CNN
- Input: 19×19 connectivity maps
- Architecture: Conv(32) → Pool → Conv(64) → Conv(128) → Channel Attention → Pool → Dense
- Multi-view support: MVSelect로 3개 view 선택 후 aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .multiview_base import MultiviewBase


class ChannelAttention(nn.Module):
    """
    논문의 Channel Attention Module
    - Shared MLP with reduction ratio = 8
    - AvgPool + MaxPool → MLP → Sigmoid
    """
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP (using Conv2d for implementation)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W] with channel attention applied
        """
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class PaperExactAttCNN(MultiviewBase):
    """
    논문 철학을 따른 Att-CNN with Multi-view support
    
    논문 구조 (19×19 connectivity matrix용):
    - 3개 Conv layers (32→64→128 filters)
    - Channel Attention (reduction=8)
    - 논문은 19×19 connectivity matrix를 직접 사용
    
    우리 적용 (224×224 RGB 이미지):
    - Input: 224×224 (connectivity heatmap visualization)
    - 논문과 동일한 layer 구성 유지
    - 224×224 입력에 맞게 stride와 pooling 조정
    - Channel Attention은 논문과 동일하게 마지막 Conv 후 적용
    
    Multi-view 적용:
    - 각 view를 개별적으로 처리
    - Max aggregation across views
    - Classification
    """
    def __init__(self, dataset):
        super(PaperExactAttCNN, self).__init__(dataset)
        
        # Dataset 정보
        self.num_class = dataset.num_class
        
        # 논문 구조: Conv(32) → Conv(64) → Conv(128) → Channel Attention
        # 224×224 입력에 맞게 stride/padding 조정
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # 224→224
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 224→112
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 112→112
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 112→112
        self.bn3 = nn.BatchNorm2d(128)
        
        # Channel Attention (reduction=8) - 논문과 동일
        self.channel_attention = ChannelAttention(128, reduction=8)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 112→56
        
        # Adaptive pooling으로 일정한 크기 유지
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # 56→4
        
        # Flatten: 4×4×128 = 2048
        flattened_size = 4 * 4 * 128  # 2048
        
        # Classifier: Dense(128) → Dense(2) - Dropout 줄임
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, self.num_class)
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward_single_view(self, x):
        """
        단일 view 처리 (논문 철학 따름)
        
        Args:
            x: [B*N, 3, 224, 224]
        Returns:
            feat: [B*N, 128, 4, 4]
        """
        # Conv1 + BN + ReLU + Pool
        x = self.conv1(x)  # [B*N, 32, 224, 224]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)  # [B*N, 32, 112, 112]
        
        # Conv2 + BN + ReLU
        x = self.conv2(x)  # [B*N, 64, 112, 112]
        x = self.bn2(x)
        x = self.relu(x)
        
        # Conv3 + BN + ReLU
        x = self.conv3(x)  # [B*N, 128, 112, 112]
        x = self.bn3(x)
        x = self.relu(x)
        
        # Channel Attention - 논문: 마지막 Conv layer 후 적용
        x = self.channel_attention(x)  # [B*N, 128, 112, 112]
        
        # MaxPool2
        x = self.pool2(x)  # [B*N, 128, 56, 56]
        
        # Adaptive pooling으로 일정한 feature 크기 유지
        x = self.adaptive_pool(x)  # [B*N, 128, 4, 4]
        
        return x
    
    def get_feat(self, imgs, M=None, down=1, visualize=False):
        """
        Multi-view feature 추출
        
        Args:
            imgs: [B, N, 3, 224, 224]
            M: view mask
            down: downsampling ratio (not used)
            visualize: True면 aggregation 전 feature map 반환
            
        Returns:
            feat: [B, N, 128, 4, 4] (visualize=True) or [B, 128, 4, 4] (aggregated)
        """
        B, N, C, H, W = imgs.shape
        
        # Reshape for batch processing
        imgs = imgs.view(B * N, C, H, W)
        
        # Forward through CNN
        feat_map = self.forward_single_view(imgs)  # [B*N, 128, 4, 4]
        
        # Reshape to [B, N, 128, 4, 4]
        _, C_feat, H_feat, W_feat = feat_map.shape
        feat_map = feat_map.view(B, N, C_feat, H_feat, W_feat)
        
        # visualize=True면 aggregation 전 feature map 반환 (MVSelect용)
        if visualize:
            return feat_map, None
        
        # Aggregation across views
        if M is not None:
            # Masked aggregation
            M_expanded = M.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1, 1]
            M_expanded = M_expanded.expand_as(feat_map)
            feat_map_masked = feat_map.masked_fill(M_expanded == 0, float('-inf'))
            feat, _ = feat_map_masked.max(dim=1)  # [B, 128, 4, 4]
        else:
            # Max pooling across all views
            feat, _ = feat_map.max(dim=1)  # [B, 128, 4, 4]
        
        return feat, None
    
    def get_output(self, overall_feat, visualize=False):
        """
        통합된 특징에서 최종 출력 생성
        
        Args:
            overall_feat: [B, 128, 4, 4]
        Returns:
            output: [B, num_class]
        """
        # Flatten: 4×4×128 = 2048
        overall_feat = overall_feat.view(overall_feat.size(0), -1)  # [B, 2048]
        
        # Classification
        output = self.classifier(overall_feat)
        
        return output
    
    def forward(self, imgs, M=None, down=1, init_prob=None, steps=0, 
                keep_cams=None, visualize=False):
        """
        Forward pass
        
        Args:
            imgs: [B, N, 3, 224, 224]
            M: view mask
            down: downsampling ratio
            init_prob: initial probability for view selection
            steps: number of view selection steps
            keep_cams: available cameras mask
            visualize: visualization flag
            
        Returns:
            output: [B, num_class]
            aux_res: auxiliary results (None for this model)
            selection_res: selection results (None for steps=0)
        """
        if steps == 0:
            # Stage 1: 모든 view 사용
            feat, _ = self.get_feat(imgs, M, down, visualize=False)
            output = self.get_output(feat, visualize=False)
            return output, None, (None, None, None, None)
        else:
            # Stage 2: MVSelect - view selection 사용
            # get_feat with visualize=True to get per-view features
            feat_map, _ = self.get_feat(imgs, M, down, visualize=True)  # [B, N, 128, 4, 4]
            
            # Aggregate for output
            if M is not None:
                M_expanded = M.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                M_expanded = M_expanded.expand_as(feat_map)
                feat_map_masked = feat_map.masked_fill(M_expanded == 0, float('-inf'))
                feat, _ = feat_map_masked.max(dim=1)
            else:
                feat, _ = feat_map.max(dim=1)
            
            output = self.get_output(feat, visualize=False)
            
            # Return with feat_map for MVSelect
            if visualize:
                return output, feat_map, (None, None, None, None)
            return output, None, (None, None, None, None)


def create_paper_exact_model(dataset):
    """
    논문 철학을 따른 모델 생성
    
    Args:
        dataset: EEG dataset
    Returns:
        model: PaperExactAttCNN instance
    """
    model = PaperExactAttCNN(dataset)
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n=== Paper-Inspired Att-CNN Model ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Input size: 224×224 (connectivity heatmap)")
    print(f"Architecture: Conv(32)→Pool→Conv(64)→Conv(128)→ChannelAtt→Pool→AdaptivePool→Dense(128)→Dense(2)")
    print(f"Feature size: 4×4×128 = 2048")
    print(f"Channel Attention: reduction=8 (논문과 동일)")
    
    return model
