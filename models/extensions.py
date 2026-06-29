# models/extensions.py
import torch
import torch.nn as nn
import torch.nn.functional as F



class SimpleChannelAttention(nn.Module):
    """通道注意力"""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 确保reduction不会导致通道数为0
        reduced_channels = max(1, in_channels // reduction)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.in_channels = in_channels  # 记录输入通道数用于动态检查

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return avg_out + max_out


class SemanticAttention(nn.Module):
    """语义注意力模块，用于处理NDVI/Sobel等单通道语义信息"""

    def __init__(self, in_channels, semantic_channels=1, reduction=16):
        super().__init__()
        self.in_channels = in_channels
        self.semantic_channels = semantic_channels

        # 语义特征处理分支
        self.semantic_conv = nn.Conv2d(semantic_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.semantic_bn = nn.BatchNorm2d(in_channels)
        self.semantic_activation = nn.Sigmoid()

        # 通道注意力
        reduced_channels = max(1, in_channels // reduction)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, features, semantic_map):
        """
        Args:
            features: 输入特征 [B, C, H, W]
            semantic_map: 语义图 [B, 1, H, W] (如NDVI或Sobel)
        """
        # 处理语义图
        semantic_weight = self.semantic_conv(semantic_map)  # [B, C, H, W]
        semantic_weight = self.semantic_bn(semantic_weight)
        semantic_weight = self.semantic_activation(semantic_weight)

        # 应用语义权重到特征
        weighted_features = features * semantic_weight

        # 应用通道注意力
        channel_att = self.channel_attention(weighted_features)
        channel_out = weighted_features * channel_att

        # 应用空间注意力
        spatial_att = self.spatial_attention(channel_out)
        spatial_out = channel_out * spatial_att

        return spatial_out



class MultiStageAttentionFusion(nn.Module):
    """多阶段注意力融合模块，用于GPT模块中"""

    def __init__(self, d_model, reduction=16):
        super().__init__()
        self.d_model = d_model
        self.ndvi_attention = SemanticAttention(d_model, semantic_channels=1, reduction=reduction)
        self.sobel_attention = SemanticAttention(d_model, semantic_channels=1, reduction=reduction)
        self.joint_attention = SemanticAttention(d_model, semantic_channels=2, reduction=reduction)

    def forward(self, rgb_features, ir_features, ndvi_map=None, sobel_map=None, fusion_type='joint'):
        """
        Args:
            rgb_features: RGB特征 [B, C, H, W]
            ir_features: IR特征 [B, C, H, W]
            ndvi_map: NDVI图 [B, 1, H, W]
            sobel_map: Sobel图 [B, 1, H, W]
            fusion_type: 'early', 'mid', 'joint'
        """
        if fusion_type == 'early':
            # 早期融合：使用NDVI引导
            if ndvi_map is not None:
                rgb_out = self.ndvi_attention(rgb_features, ndvi_map)
                ir_out = self.ndvi_attention(ir_features, ndvi_map)
            else:
                rgb_out, ir_out = rgb_features, ir_features
        elif fusion_type == 'mid':
            # 中期融合：使用Sobel引导
            if sobel_map is not None:
                rgb_out = self.sobel_attention(rgb_features, sobel_map)
                ir_out = self.sobel_attention(ir_features, sobel_map)
            else:
                rgb_out, ir_out = rgb_features, ir_features
        else:  # joint
            # 联合融合：使用NDVI和Sobel
            if ndvi_map is not None and sobel_map is not None:
                combined_semantic = torch.cat([ndvi_map, sobel_map], dim=1)
                rgb_out = self.joint_attention(rgb_features, combined_semantic)
                ir_out = self.joint_attention(ir_features, combined_semantic)
            else:
                rgb_out, ir_out = rgb_features, ir_features

        return rgb_out, ir_out

class SimpleSemanticAttention(nn.Module):
    """简单的语义注意力模块，使用单通道语义图生成注意力权重"""

    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(1, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, features, semantic_map):
        """
        Args:
            features: 输入特征 [B, C, H, W]
            semantic_map: 语义图 [B, 1, H, W]
        """
        # 通道注意力
        channel_weights = self.channel_attention(semantic_map)

        # 空间注意力
        spatial_weights = self.spatial_attention(semantic_map)

        # 结合通道和空间注意力
        weights = channel_weights * spatial_weights

        # 应用注意力权重
        attended_features = features * weights

        return attended_features