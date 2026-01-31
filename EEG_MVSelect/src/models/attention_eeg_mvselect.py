"""
Attention 강화 EEG Multi-View Selection Model
- ResNet50 백본 (더 깊은 네트워크)
- CBAM (Channel + Spatial Attention) 추가
- Multi-View Attention 추가
- View Selection Attention으로 더 나은 뷰 선택
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .multiview_base import MultiviewBase
from .mvselect import CamSelect
from .attention_modules import CBAM, MultiViewAttention, ViewSelectionAttention


class AttentionEEGMVSelect(MultiviewBase):
    """
    Attention 메커니즘이 강화된 EEG Multi-View Selection 모델
    
    개선 사항:
    1. ResNet50으로 더 깊은 feature 학습
    2. CBAM으로 공간적/채널별 중요 feature 강조
    3. Multi-View Attention으로 뷰 간 관계 학습
    4. View Selection Attention으로 더 나은 뷰 선택
    """
    
    def __init__(self, dataset, arch='resnet50', aggregation='max', pretrained=True, 
                 use_cbam=True, use_mv_attention=True, use_vs_attention=True):
        super().__init__(dataset, aggregation)
        
        self.use_cbam = use_cbam
        self.use_mv_attention = use_mv_attention
        self.use_vs_attention = use_vs_attention
        
        # 백본 네트워크 설정
        if arch == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            # ResNet의 마지막 layer까지 사용 (layer4까지)
            self.base = nn.Sequential(*list(base_model.children())[:-2])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            base_dim = 2048  # ResNet50의 출력 차원
            
        elif arch == 'resnet101':
            base_model = models.resnet101(pretrained=pretrained)
            self.base = nn.Sequential(*list(base_model.children())[:-2])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            base_dim = 2048
            
        elif arch == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            self.base = nn.Sequential(*list(base_model.children())[:-2])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            base_dim = 512
            
        elif arch == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            self.base = nn.Sequential(*list(base_model.children())[:-2])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            base_dim = 512
            
        else:
            raise ValueError(f"Unsupported architecture: {arch}. "
                           f"Choose from ['resnet18', 'resnet34', 'resnet50', 'resnet101']")
        
        self.arch = arch
        self.base_dim = base_dim
        
        # CBAM Attention 추가 (feature map에 적용)
        if self.use_cbam:
            self.cbam = CBAM(base_dim, reduction=16, kernel_size=7)
            print(f"  ✓ CBAM Attention enabled (Channel + Spatial)")
        
        # Multi-View Attention 추가 (view 간 관계 학습)
        if self.use_mv_attention:
            self.mv_attention = MultiViewAttention(
                feature_dim=base_dim,
                num_views=dataset.num_cam,
                num_heads=8
            )
            print(f"  ✓ Multi-View Attention enabled (8 heads)")
        
        # View Selection Attention 추가 (더 나은 뷰 선택)
        if self.use_vs_attention:
            self.vs_attention = ViewSelectionAttention(
                feature_dim=base_dim,
                num_views=dataset.num_cam
            )
            print(f"  ✓ View Selection Attention enabled")
        
        # 분류기 (ADHD vs Control)
        # 과적합 방지를 위해 더 강한 regularization 적용
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),  # 0.5 → 0.6 증가
            nn.Linear(base_dim, base_dim // 4),  # // 2 → // 4 (더 작은 hidden layer)
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(base_dim // 4),
            nn.Dropout(0.5),  # 0.3 → 0.5 증가
            nn.Linear(base_dim // 4, dataset.num_class)
        )
        
        # View Selection 모듈 (개선된 버전)
        self.select_module = CamSelect(
            num_cam=dataset.num_cam,
            hidden_dim=base_dim,
            kernel_size=1,
            aggregation=aggregation
        )
        
        print(f"Attention-Enhanced EEG-MVSelect Model Initialized:")
        print(f"  Architecture: {arch}")
        print(f"  Base dimension: {base_dim}")
        print(f"  Number of views: {dataset.num_cam}")
        print(f"  Number of classes: {dataset.num_class}")
        print(f"  Aggregation: {aggregation}")
        print(f"  Pretrained: {pretrained}")
    
    def get_feat(self, imgs, M=None, down=1, visualize=False):
        """
        이미지에서 feature 추출 (기존 모델과 호환되는 인터페이스)
        
        Args:
            imgs: [B, num_views, C, H, W] 또는 [B*num_views, C, H, W]
            M: view mask (사용할 뷰 선택)
            down: downsampling 비율
            visualize: 시각화 여부
            
        Returns:
            feat: [B, N, base_dim, H', W'] feature maps (MVSelect용) 또는 [B, base_dim] aggregated feature
            aux_res: None (compatibility)
        """
        # MVSelect 모드인지 확인 (M이 None이고 visualize가 False면 MVSelect 모드로 추정)
        # Trainer가 get_feat를 호출하는 경우는 MVSelect 모드
        if M is None and not visualize:
            # MVSelect를 위해 feature map 반환
            feat_map, aux_res = self.get_feat_before_pooling(imgs, down)
            return feat_map, aux_res
        
        # 일반 모드에서는 aggregated feature 반환
        B, N, C, H, W = imgs.shape if len(imgs.shape) == 5 else (imgs.shape[0] // self.num_cam, self.num_cam, *imgs.shape[1:])
        
        # Reshape for batch processing
        if len(imgs.shape) == 5:
            imgs = imgs.view(B * N, C, H, W)
        
        # Downsampling
        if down > 1:
            imgs = F.interpolate(imgs, scale_factor=1/down, mode='bilinear', align_corners=False)
        
        # 백본 네트워크로 feature 추출
        feat_map = self.base(imgs)  # [B*N, base_dim, H', W']
        
        # CBAM Attention 적용
        if self.use_cbam:
            feat_map = self.cbam(feat_map)  # [B*N, base_dim, H', W']
        
        # Global average pooling
        feat = self.avgpool(feat_map)  # [B*N, base_dim, 1, 1]
        feat = feat.view(B, N, self.base_dim)  # [B, N, base_dim]
        
        # Multi-View Attention 적용
        if self.use_mv_attention:
            # M이 있으면 mask로 사용
            mask = M if M is not None else None
            feat = self.mv_attention(feat, mask)  # [B, N, base_dim]
        
        # View Selection Attention 적용 (view importance 학습)
        view_importance = None
        if self.use_vs_attention:
            feat, view_importance = self.vs_attention(feat)  # [B, N, base_dim], [B, N]
        
        # View aggregation
        if M is not None:
            # Masked views만 사용
            M_expanded = M.unsqueeze(-1).expand_as(feat)  # [B, N, base_dim]
            
            # View importance와 mask를 모두 고려
            if view_importance is not None:
                # importance score를 mask와 결합
                weights = M * F.softmax(view_importance, dim=1)
                weights = weights.unsqueeze(-1).expand_as(feat)  # [B, N, base_dim]
                feat = (feat * weights).sum(dim=1)  # Weighted sum
            else:
                if self.aggregation == 'max':
                    feat = feat.masked_fill(M_expanded == 0, float('-inf'))
                    feat, _ = feat.max(dim=1)  # [B, base_dim]
                else:  # mean
                    feat = (feat * M_expanded).sum(dim=1) / M.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            # 모든 뷰 사용
            if view_importance is not None:
                # importance score 기반 weighted aggregation
                weights = F.softmax(view_importance, dim=1).unsqueeze(-1)  # [B, N, 1]
                feat = (feat * weights).sum(dim=1)  # [B, base_dim]
            else:
                if self.aggregation == 'max':
                    feat, _ = feat.max(dim=1)
                else:  # mean
                    feat = feat.mean(dim=1)
        
        if visualize:
            return feat, feat_map
        return feat, None  # 호환성을 위해 None 추가
    
    def forward(self, imgs, M=None, down=1, init_prob=None, steps=0, 
                keep_cams=None, visualize=False):
        """
        Forward pass (호환성을 위해 기존 모델과 동일한 인터페이스)
        
        Args:
            imgs: [B, N, C, H, W] input images
            M: [B, N] view mask
            down: downsampling ratio
            init_prob: initial probability for view selection
            steps: number of view selection steps
            keep_cams: available cameras mask
            visualize: visualization flag
            
        Returns:
            output: [B, num_class] classification logits
            aux_res: None (no auxiliary output)
            select_res: (log_probs, values, actions, entropies) if steps > 0
        """
        B, N, C, H, W = imgs.shape
        
        # MVSelect 모드 (view selection)
        if steps > 0 and M is None:
            # View selection을 통해 일부 뷰만 선택
            # MultiviewBase의 select_module 사용
            imgs_feat, _ = self.get_feat_before_pooling(imgs, down)
            
            # Select module로 뷰 선택
            log_probs, values, actions, entropies, M = self.select_module(
                imgs_feat, init_prob, steps, keep_cams
            )
            select_res = (log_probs, values, actions, entropies)
        else:
            select_res = (None, None, None, None)
        
        # Feature extraction with selected views
        feat = self.get_feat(imgs, M, down)  # [B, base_dim]
        
        # Classification
        output = self.classifier(feat)  # [B, num_class]
        
        return output, None, select_res
    
    def get_output(self, overall_feat, visualize=False):
        """
        통합된 특징에서 최종 출력 생성
        
        Args:
            overall_feat: [B, base_dim, H, W] 형태의 통합 특징
            visualize: 시각화 플래그
            
        Returns:
            output: [B, num_class] 형태의 분류 결과
        """
        # Global average pooling
        overall_feat = self.avgpool(overall_feat)  # [B, base_dim, 1, 1]
        overall_feat = torch.flatten(overall_feat, 1)  # [B, base_dim]
        
        # 분류
        output = self.classifier(overall_feat)
        
        return output
    
    def get_feat_before_pooling(self, imgs, down=1):
        """
        Pooling 전 feature map 추출 (view selection을 위해)
        
        Args:
            imgs: [B, N, C, H, W]
            down: downsampling ratio
            
        Returns:
            feat_map: [B, N, base_dim, H', W']
            aux_res: None
        """
        B, N, C, H, W = imgs.shape
        
        # Reshape for batch processing
        imgs = imgs.view(B * N, C, H, W)
        
        # Downsampling
        if down > 1:
            imgs = F.interpolate(imgs, scale_factor=1/down, mode='bilinear', align_corners=False)
        
        # 백본 네트워크로 feature 추출
        feat_map = self.base(imgs)  # [B*N, base_dim, H', W']
        
        # CBAM Attention 적용
        if self.use_cbam:
            feat_map = self.cbam(feat_map)  # [B*N, base_dim, H', W']
        
        # Reshape to [B, N, C, H, W]
        _, C_feat, H_feat, W_feat = feat_map.shape
        feat_map = feat_map.view(B, N, C_feat, H_feat, W_feat)
        
        return feat_map, None
    
    def get_select_loss(self, logits, target, pred_conf, action, reward):
        """
        View selection을 위한 강화학습 loss
        
        Args:
            logits: [B, num_class] classification logits
            target: [B] ground truth labels
            pred_conf: [B] prediction confidence
            action: [B, N] selected actions
            reward: [B] rewards
            
        Returns:
            total_loss: scalar loss
            cls_loss: classification loss
            rl_loss: reinforcement learning loss
        """
        # Classification loss
        cls_loss = F.cross_entropy(logits, target)
        
        # RL loss (policy gradient)
        if action is not None and reward is not None:
            # Normalize rewards
            reward = (reward - reward.mean()) / (reward.std() + 1e-8)
            
            # Policy gradient loss
            log_probs = F.log_softmax(pred_conf, dim=-1)
            selected_log_probs = (log_probs * action).sum(dim=-1)
            rl_loss = -(selected_log_probs * reward).mean()
        else:
            rl_loss = torch.tensor(0.0, device=logits.device)
        
        return cls_loss, rl_loss


def create_attention_eeg_mvselect_model(dataset, arch='resnet50', aggregation='max', 
                                       pretrained=True, use_cbam=True, 
                                       use_mv_attention=True, use_vs_attention=True):
    """
    Attention 강화 EEG MVSelect 모델 생성
    
    Args:
        dataset: EEGConnectivityDataset
        arch: 백본 아키텍처 (resnet18/34/50/101)
        aggregation: view aggregation 방법 (max/mean)
        pretrained: ImageNet pretrained weights 사용 여부
        use_cbam: CBAM attention 사용 여부
        use_mv_attention: Multi-view attention 사용 여부
        use_vs_attention: View selection attention 사용 여부
        
    Returns:
        model: AttentionEEGMVSelect 인스턴스
    """
    model = AttentionEEGMVSelect(
        dataset=dataset,
        arch=arch,
        aggregation=aggregation,
        pretrained=pretrained,
        use_cbam=use_cbam,
        use_mv_attention=use_mv_attention,
        use_vs_attention=use_vs_attention
    )
    return model
