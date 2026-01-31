"""
Attention 메커니즘 모듈
- Spatial Attention: 뇌 연결성 이미지의 중요 영역 학습
- Channel Attention: 중요 feature 채널 학습
- Multi-View Attention: View 간 관계 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CBAM 스타일)
    어떤 feature channel이 중요한지 학습
    """
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
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
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (CBAM 스타일)
    이미지의 어떤 공간 영역이 중요한지 학습
    뇌 연결성 매트릭스에서 중요한 연결 패턴을 찾는데 유용
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W] with spatial attention applied
        """
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Concatenate along channel dimension
        x_cat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        
        # Spatial attention map
        attention = self.sigmoid(self.conv(x_cat))  # [B, 1, H, W]
        
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    Channel + Spatial Attention을 순차적으로 적용
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W] with CBAM applied
        """
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class MultiViewAttention(nn.Module):
    """
    Multi-View Attention Module
    여러 뷰 간의 관계를 학습하고 중요한 뷰에 더 집중
    """
    def __init__(self, feature_dim, num_views, num_heads=4):
        super(MultiViewAttention, self).__init__()
        self.num_views = num_views
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        self.head_dim = feature_dim // num_heads
        
        # Query, Key, Value projections
        self.q_linear = nn.Linear(feature_dim, feature_dim)
        self.k_linear = nn.Linear(feature_dim, feature_dim)
        self.v_linear = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, x, mask=None):
        """
        Multi-head attention across views
        
        Args:
            x: [B, N, D] where N is num_views, D is feature_dim
            mask: [B, N] binary mask (1 for valid views, 0 for invalid)
        Returns:
            out: [B, N, D] with multi-view attention applied
        """
        B, N, D = x.shape
        
        # Linear projections and reshape for multi-head
        Q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        K = self.k_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        V = self.v_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, N, N]
        
        # Apply mask if provided
        if mask is not None:
            # mask: [B, N] -> [B, 1, 1, N]
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [B, H, N, N]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # [B, H, N, d]
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]
        
        # Output projection
        output = self.out_proj(context)
        
        # Residual connection + Layer norm
        output = self.layer_norm(x + self.dropout(output))
        
        return output


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    Channel-wise feature recalibration
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W] with SE applied
        """
        B, C, _, _ = x.size()
        # Squeeze
        y = self.squeeze(x).view(B, C)
        # Excitation
        y = self.excitation(y).view(B, C, 1, 1)
        # Scale
        return x * y.expand_as(x)


class ViewSelectionAttention(nn.Module):
    """
    View Selection을 위한 Attention
    각 뷰의 중요도를 학습하여 view selection에 활용
    """
    def __init__(self, feature_dim, num_views):
        super(ViewSelectionAttention, self).__init__()
        self.num_views = num_views
        
        # View importance scoring
        self.importance_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, 1)
        )
        
        # View feature refinement
        self.refine_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, x):
        """
        Args:
            x: [B, N, D] view features
        Returns:
            refined_features: [B, N, D]
            importance_scores: [B, N] importance scores for each view
        """
        B, N, D = x.shape
        
        # Compute importance scores
        importance_scores = self.importance_net(x).squeeze(-1)  # [B, N]
        importance_weights = F.softmax(importance_scores, dim=1).unsqueeze(-1)  # [B, N, 1]
        
        # Refine features with importance weights
        refined = self.refine_net(x)
        refined = self.layer_norm(x + refined * importance_weights)
        
        return refined, importance_scores.squeeze(-1)
