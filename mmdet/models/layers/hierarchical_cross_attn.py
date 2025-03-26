import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union
from torch import Tensor

from mmengine.model import BaseModule
from mmdet.registry import MODELS


class ScaleLevelAttention(nn.Module):
    """单一尺度级别的跨模态注意力模块
    
    在特定尺度级别上计算视觉特征与文本特征之间的注意力。
    
    Args:
        embed_dims (int): 特征嵌入维度
        num_heads (int): 多头注意力的头数
        dropout (float): 注意力dropout率
    """
    def __init__(self, 
                 embed_dims: int, 
                 num_heads: int = 8, 
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.dropout = dropout
        
        # 投影层
        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        
        # 输出层规范化和前馈网络
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dims * 4, embed_dims),
            nn.Dropout(dropout)
        )
        
    def forward(self, visual_feat: Tensor, text_feat: Tensor, 
                visual_mask: Optional[Tensor] = None,
                text_mask: Optional[Tensor] = None) -> Tensor:
        """前向传播
        
        Args:
            visual_feat (Tensor): 视觉特征，形状为 (B, H*W, C)
            text_feat (Tensor): 文本特征，形状为 (B, L, C)
            visual_mask (Tensor, optional): 视觉特征掩码
            text_mask (Tensor, optional): 文本特征掩码
            
        Returns:
            Tensor: 注意力增强的视觉特征，形状为 (B, H*W, C)
        """
        B, N, C = visual_feat.shape
        L = text_feat.shape[1]
        
        # 将视觉特征作为查询，文本特征作为键和值
        q = self.q_proj(visual_feat).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(text_feat).reshape(B, L, self.num_heads, C // self.num_heads)
        v = self.v_proj(text_feat).reshape(B, L, self.num_heads, C // self.num_heads)
        
        # 调整维度顺序 (B, num_heads, N/L, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        # 计算注意力权重
        attn_weight = torch.matmul(q, k.transpose(-2, -1)) / (C // self.num_heads) ** 0.5
        
        # 应用掩码（如果提供）
        if text_mask is not None:
            # 扩展掩码以适应多头注意力格式
            text_mask = text_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            attn_weight = attn_weight.masked_fill(~text_mask, float('-inf'))
        
        # 归一化注意力权重
        attn_weight = F.softmax(attn_weight, dim=-1)
        attn_weight = F.dropout(attn_weight, p=self.dropout, training=self.training)
        
        # 应用注意力权重
        out = torch.matmul(attn_weight, v)  # (B, num_heads, N, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)  # (B, N, C)
        out = self.out_proj(out)
        
        # 残差连接与LayerNorm
        visual_feat = self.norm1(visual_feat + out)
        visual_feat = self.norm2(visual_feat + self.ffn(visual_feat))
        
        return visual_feat


@MODELS.register_module()
class HierarchicalCrossAttention(BaseModule):
    """层次化多尺度跨模态注意力模块
    
    对多个尺度级别的视觉特征分别应用跨模态注意力，
    并在不同尺度间实现特征交互。
    
    Args:
        embed_dims (int): 特征嵌入维度
        num_levels (int): 特征金字塔中的层级数
        num_heads (int): 多头注意力的头数
        dropout (float): dropout率
        norm_cfg (dict): 归一化层配置
        init_cfg (dict): 初始化配置
    """
    def __init__(self,
                 embed_dims: int,
                 num_levels: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 cross_level_interaction: bool = True,
                 return_separate_levels: bool = False,
                 norm_cfg: dict = dict(type='LN'),
                 init_cfg: dict = None):
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.dropout = dropout
        self.cross_level_interaction = cross_level_interaction
        self.return_separate_levels = return_separate_levels
        
        # 为每个尺度级别创建一个跨模态注意力模块
        self.scale_attentions = nn.ModuleList([
            ScaleLevelAttention(embed_dims, num_heads, dropout)
            for _ in range(num_levels)
        ])
        
        # 尺度级别间的交互模块（如果启用）
        if cross_level_interaction:
            self.level_interactions = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dims * 2, embed_dims),
                    nn.LayerNorm(embed_dims),
                    nn.ReLU(inplace=True)
                ) for _ in range(num_levels - 1)
            ])
            
            self.top_down_projections = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(embed_dims, embed_dims, kernel_size=1),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                ) for _ in range(num_levels - 1)
            ])
            
            self.bottom_up_projections = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=2, padding=1)
                ) for _ in range(num_levels - 1)
            ])
        
        # 文本特征投影
        self.text_projections = nn.ModuleList([
            nn.Linear(embed_dims, embed_dims)
            for _ in range(num_levels)
        ])
        
        # 输出特征融合
        self.fusion = nn.Sequential(
            nn.Linear(embed_dims * num_levels, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True)
        ) if not return_separate_levels else None
    
    def forward(self, 
                multi_scale_feats: List[Tensor], 
                text_feat: Tensor,
                text_mask: Optional[Tensor] = None) -> Union[Tensor, List[Tensor]]:
        """前向传播
        
        Args:
            multi_scale_feats (List[Tensor]): 多尺度视觉特征列表，每个张量形状为
                (B, C, Hi, Wi)，索引从细粒度到粗粒度排序
            text_feat (Tensor): 文本特征，形状为 (B, L, C)
            text_mask (Tensor, optional): 文本特征掩码，形状为 (B, L)
            
        Returns:
            Union[Tensor, List[Tensor]]: 
                如果 return_separate_levels=False，返回融合后的特征，形状为 (B, H*W, C)
                否则返回处理后的多尺度特征列表
        """
        assert len(multi_scale_feats) == self.num_levels, \
            f"输入特征层级数 ({len(multi_scale_feats)}) 与预期数量 ({self.num_levels}) 不一致"
        
        batch_size = multi_scale_feats[0].shape[0]
        
        # 保存处理前的原始特征尺寸
        original_shapes = []
        flattened_feats = []
        
        # 展平每个尺度的特征
        for i, feat in enumerate(multi_scale_feats):
            B, C, H, W = feat.shape
            original_shapes.append((H, W))
            # 重塑为序列形式 (B, C, H, W) -> (B, H*W, C)
            flattened_feat = feat.flatten(2).permute(0, 2, 1)
            flattened_feats.append(flattened_feat)
        
        # 为每个尺度级别投影文本特征
        projected_text_feats = [
            proj(text_feat) for proj in self.text_projections
        ]
        
        # 对每个尺度级别应用注意力
        attended_feats = []
        for i, feat in enumerate(flattened_feats):
            attended_feat = self.scale_attentions[i](
                feat, projected_text_feats[i], text_mask=text_mask
            )
            attended_feats.append(attended_feat)
            
        # 如果启用尺度级别间交互
        if self.cross_level_interaction:
            # 首先将展平的特征重塑回空间维度
            spatial_feats = []
            for i, feat in enumerate(attended_feats):
                H, W = original_shapes[i]
                # (B, H*W, C) -> (B, C, H, W)
                spatial_feat = feat.permute(0, 2, 1).reshape(batch_size, self.embed_dims, H, W)
                spatial_feats.append(spatial_feat)
            
            # 自顶向下的路径 (粗粒度 -> 细粒度)
            top_down_feats = [spatial_feats[-1]]  # 从最粗粒度开始
            for i in range(self.num_levels - 2, -1, -1):
                # 上一层特征上采样并与当前层融合
                td_feat = self.top_down_projections[self.num_levels - 2 - i](top_down_feats[-1])
                top_down_feats.append(td_feat + spatial_feats[i])
            top_down_feats = top_down_feats[::-1]  # 反转列表，使其从细粒度到粗粒度
            
            # 自底向上的路径 (细粒度 -> 粗粒度)
            bottom_up_feats = [top_down_feats[0]]  # 从最细粒度开始
            for i in range(self.num_levels - 1):
                # 当前层特征下采样并与下一层融合
                bu_feat = self.bottom_up_projections[i](bottom_up_feats[-1])
                bottom_up_feats.append(bu_feat + top_down_feats[i + 1])
            
            # 展平回序列形式并融合特征
            enhanced_feats = []
            for i, feat in enumerate(bottom_up_feats):
                H, W = original_shapes[i]
                # (B, C, H, W) -> (B, H*W, C)
                flat_feat = feat.flatten(2).permute(0, 2, 1)
                
                # 与原始特征融合
                enhanced_feat = torch.cat([flat_feat, attended_feats[i]], dim=-1)
                enhanced_feat = self.level_interactions[i](enhanced_feat) if i < self.num_levels - 1 else flat_feat
                enhanced_feats.append(enhanced_feat)
        else:
            enhanced_feats = attended_feats
        
        # 返回多层级特征或融合后的单一特征
        if self.return_separate_levels:
            return enhanced_feats
        else:
            # 将不同尺度的特征插值到最细粒度
            finest_h, finest_w = original_shapes[0]
            aligned_feats = []
            
            for i, feat in enumerate(enhanced_feats):
                if i == 0:  # 最细粒度层级无需插值
                    aligned_feats.append(feat)
                else:
                    # 将特征重塑回空间维度
                    B, N, C = feat.shape
                    h, w = original_shapes[i]
                    spatial_feat = feat.permute(0, 2, 1).reshape(B, C, h, w)
                    
                    # 插值到最细粒度
                    upsampled_feat = F.interpolate(
                        spatial_feat, size=(finest_h, finest_w), 
                        mode='bilinear', align_corners=False
                    )
                    
                    # 重塑回序列形式
                    aligned_feat = upsampled_feat.flatten(2).permute(0, 2, 1)
                    aligned_feats.append(aligned_feat)
            
            # 沿特征维度拼接并融合
            concat_feat = torch.cat(aligned_feats, dim=-1)
            fused_feat = self.fusion(concat_feat)
            
            return fused_feat 