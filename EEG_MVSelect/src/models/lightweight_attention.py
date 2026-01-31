"""
논문 스타일의 경량 Attention 모델
- Channel Attention만 사용 (CBAM의 절반)
- 더 얕은 네트워크 (과적합 방지)
- 논문과 유사한 구조
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .multiview_base import MultiviewBase


class ChannelAttentionOnly(nn.Module):
    """
    논문 스타일의 Channel Attention (CBAM의 절반)
    Spatial Attention은 제외
    """
    def __init__(self, channels, reduction=8):
        super(ChannelAttentionOnly, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
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


class LightweightAttentionCNN(MultiviewBase):
    """
    논문 스타일의 경량 Attention CNN
    - 더 얕은 네트워크 (3개 Conv layer)
    - Channel Attention만 사용
    - 과적합 방지를 위한 단순한 구조
    """
    
    def __init__(self, dataset, pretrained=False):
        super().__init__(dataset, aggregation='max')
        
        # 3-layer CNN (논문 스타일)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Channel Attention (논문과 동일)
        self.channel_attention = ChannelAttentionOnly(128, reduction=8)
        
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Adaptive pooling for variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Calculate flattened size
        # After all conv/pool: [B, 128, 4, 4] = 128 * 4 * 4 = 2048
        flattened_size = 128 * 4 * 4
        
        # Classifier (더 단순하게)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, dataset.num_class)
        )
        
        print(f"Lightweight Attention CNN Initialized (논문 스타일):")
        print(f"  Architecture: 3-layer CNN + Channel Attention")
        print(f"  Number of views: {dataset.num_cam}")
        print(f"  Number of classes: {dataset.num_class}")
        print(f"  Total parameters: ~1M (vs ResNet50 25M)")
    
    def forward_single_view(self, x):
        """
        단일 뷰 이미지 처리
        
        Args:
            x: [B*N, C, H, W]
        Returns:
            feat: [B*N, 128, H', W']
        """
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Channel Attention (논문의 핵심)
        x = self.channel_attention(x)
        
        # Final pooling
        x = self.pool2(x)
        x = self.adaptive_pool(x)
        
        return x
    
    def get_feat(self, imgs, M=None, down=1, visualize=False):
        """
        Feature 추출
        
        Args:
            imgs: [B, N, C, H, W]
            M: view mask
            down: downsampling ratio
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
        # Flatten
        overall_feat = overall_feat.view(overall_feat.size(0), -1)  # [B, 128*4*4]
        
        # Classification
        output = self.classifier(overall_feat)
        
        return output
    
    def forward(self, imgs, M=None, down=1, init_prob=None, steps=0, 
                keep_cams=None, visualize=False):
        """
        Forward pass
        
        Args:
            imgs: [B, N, C, H, W]
            M: view mask
            down: downsampling ratio
            init_prob: initial probability for view selection
            steps: number of view selection steps
            keep_cams: available cameras mask
            visualize: visualization flag
            
        Returns:
            output: [B, num_class]
            aux_res: None
            select_res: (log_probs, values, actions, entropies) if steps > 0
        """
        B, N, C, H, W = imgs.shape
        
        # MVSelect 모드 (view selection)
        if steps > 0:
            # View selection을 통해 일부 뷰만 선택
            imgs_feat, _ = self.get_feat(imgs, M=None, down=down, visualize=True)  # Get [B, N, C, H, W]
            
            # Select module로 뷰 선택
            log_probs, values, actions, entropies, M = self.select_module(
                imgs_feat, init_prob, steps, keep_cams
            )
            select_res = (log_probs, values, actions, entropies)
            
            # 선택된 뷰로 aggregation
            feat, _ = self.get_feat(imgs, M=M, down=down)  # [B, 128, 4, 4]
        else:
            select_res = (None, None, None, None)
            # 모든 뷰 사용 (aggregation 수행)
            feat, _ = self.get_feat(imgs, M=None, down=down, visualize=False)  # [B, 128, 4, 4]
        
        # Flatten and classify
        feat_flat = feat.view(feat.size(0), -1)  # [B, 2048]
        output = self.classifier(feat_flat)
        
        return output, None, select_res


def create_lightweight_attention_model(dataset, pretrained=False):
    """
    논문 스타일의 경량 Attention 모델 생성
    
    Args:
        dataset: EEGConnectivityDataset
        pretrained: 사용 안 함 (간단한 CNN)
        
    Returns:
        model: LightweightAttentionCNN 인스턴스
    """
    model = LightweightAttentionCNN(dataset, pretrained=False)
    return model
