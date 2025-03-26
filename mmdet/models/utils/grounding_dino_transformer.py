 # 首先检查文件是否存在，如果不存在，我们创建这个文件
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
from torch import Tensor
from mmcv.cnn import Linear
from mmengine.model import BaseModule
from mmdet.registry import MODELS
# 导入我们刚刚创建的模块
from mmdet.models.layers import HierarchicalCrossAttention



class GroundingDinoTransformerEncoder(BaseModule):
    """Grounding DINO Transformer Encoder with层次化多尺度跨模态注意力机制.
    
    基于原始DINO的编码器，增加了层次化多尺度跨模态注意力机制，
    能够更好地将文本特征与不同空间分辨率的视觉特征对齐。
    
    Args:
        embed_dims (int): 特征嵌入维度
        num_layers (int): 编码器层数
        num_feature_levels (int): 特征层级数，默认为4
        num_heads (int): 多头注意力的头数
        feedforward_channels (int): FFN隐藏层通道数
        dropout (float): Dropout率
        cross_attn_cfg (dict): 跨模态注意力配置
        ffn_dropout (float): FFN的dropout率
        operation_order (tuple[str]): 编码器中操作的顺序
        norm_cfg (dict): 规范化配置
        init_cfg (dict): 初始化配置
    """
    def __init__(self,
                 embed_dims: int = 256,
                 num_layers: int = 6,
                 num_feature_levels: int = 4,
                 num_heads: int = 8,
                 feedforward_channels: int = 1024,
                 dropout: float = 0.1,
                 cross_attn_cfg: Optional[dict] = None,
                 ffn_dropout: float = 0.1,
                 operation_order: tuple = ('self_attn', 'norm', 'ffn', 'norm'),
                 norm_cfg: dict = dict(type='LN'),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)
        
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        
        # 创建标准Transformer编码器层
        encoder_layer = TransformerEncoderLayer(
            embed_dims=embed_dims,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            norm_cfg=norm_cfg)
        
        self.encoder_layers = nn.ModuleList(
            [encoder_layer for _ in range(num_layers)])
        
        # 创建层次化多尺度跨模态注意力模块
        self.cross_attn_cfg = cross_attn_cfg or dict(
            embed_dims=embed_dims,
            num_levels=num_feature_levels,
            num_heads=num_heads,
            dropout=dropout,
            cross_level_interaction=True,
            return_separate_levels=True)
        
        self.hierarchical_cross_attn = MODELS.build(
            dict(type='HierarchicalCrossAttention', **self.cross_attn_cfg))
        
        # 用于处理经过跨模态注意力后的特征的投影层
        self.level_embed = nn.Parameter(
            torch.Tensor(num_feature_levels, embed_dims))
        self.level_projs = nn.ModuleList([
            nn.Linear(embed_dims, embed_dims) 
            for _ in range(num_feature_levels)
        ])
        
        # 初始化参数
        nn.init.normal_(self.level_embed)
    
    def forward(self,
                multi_level_feats: List[Tensor],
                multi_level_masks: List[Tensor],
                multi_level_pos_embeds: List[Tensor],
                text_feat: Optional[Tensor] = None,
                text_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """前向传播函数
        
        Args:
            multi_level_feats (List[Tensor]): 多层级特征，每个张量形状为
                (bs, c, h, w)
            multi_level_masks (List[Tensor]): 多层级掩码，每个张量形状为
                (bs, h, w)
            multi_level_pos_embeds (List[Tensor]): 多层级位置编码，每个张量形状为
                (bs, c, h, w)
            text_feat (Tensor, optional): 文本特征，形状为 (bs, l, c)
            text_mask (Tensor, optional): 文本掩码，形状为 (bs, l)
                
        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - 融合后的特征，形状为 (bs, sum(h*w), c)
                - 融合后的掩码，形状为 (bs, sum(h*w))
                - 融合后的位置编码，形状为 (bs, sum(h*w), c)
                - 空间形状，形状为 (num_levels, 2)
                - 各级别起始索引，形状为 (num_levels,)
        """
        feat_flatten = []
        mask_flatten = []
        pos_embed_flatten = []
        spatial_shapes = []
        
        # 如果提供了文本特征，应用层次化多尺度跨模态注意力
        if text_feat is not None:
            # 应用跨模态注意力，返回多层级特征
            enhanced_feats = self.hierarchical_cross_attn(
                multi_level_feats, text_feat, text_mask)
            
            # 更新多层级特征
            for i in range(self.num_feature_levels):
                B, N, C = enhanced_feats[i].shape
                H, W = multi_level_masks[i].shape[1:]
                
                # 将增强特征重塑回空间形式
                enhanced_feat = enhanced_feats[i].permute(0, 2, 1).reshape(B, C, H, W)
                
                # 用增强特征替换原始特征
                multi_level_feats[i] = enhanced_feat
        
        # 处理多层级特征，并展平为序列
        for i in range(self.num_feature_levels):
            bs, c, h, w = multi_level_feats[i].shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            
            # 展平特征、掩码和位置编码
            feat = multi_level_feats[i].flatten(2).transpose(1, 2)  # (bs, h*w, c)
            mask = multi_level_masks[i].flatten(1)  # (bs, h*w)
            pos_embed = multi_level_pos_embeds[i].flatten(2).transpose(1, 2)  # (bs, h*w, c)
            
            # 添加层级嵌入，增强不同尺度特征的区分性
            level_embed = self.level_embed[i].view(1, 1, -1)
            level_pos_embed = level_embed.expand(bs, h*w, c)
            pos_embed = pos_embed + level_pos_embed
            
            # 对层级特征进行投影，确保不同尺度特征的一致性
            feat = self.level_projs[i](feat)
            
            feat_flatten.append(feat)
            mask_flatten.append(mask)
            pos_embed_flatten.append(pos_embed)
        
        # 将所有层级特征拼接在一起
        feat_flatten = torch.cat(feat_flatten, 1)  # (bs, sum(h*w), c)
        mask_flatten = torch.cat(mask_flatten, 1)  # (bs, sum(h*w))
        pos_embed_flatten = torch.cat(pos_embed_flatten, 1)  # (bs, sum(h*w), c)
        
        # 计算各级别起始索引
        level_start_index = torch.cat((
            feat_flatten.new_zeros((1,), dtype=torch.long),
            torch.tensor([prod(shape) for shape in spatial_shapes],
                        dtype=torch.long, device=feat_flatten.device).cumsum(0)))
        
        # 空间形状张量化
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        
        # 对拼接后的特征应用Transformer编码器层
        for enc_layer in self.encoder_layers:
            feat_flatten = enc_layer(
                query=feat_flatten,
                key=None,
                value=None,
                query_pos=pos_embed_flatten,
                query_key_padding_mask=mask_flatten)
        
        return feat_flatten, mask_flatten, pos_embed_flatten, spatial_shapes, level_start_index


# 辅助函数
def prod(x):
    """计算可迭代对象中所有元素的乘积"""
    p = 1
    for i in x:
        p *= i
    return p


# 标准Transformer编码器层的实现
class TransformerEncoderLayer(BaseModule):
    """标准的Transformer编码器层
    
    包含自注意力和前馈网络。
    
    Args:
        embed_dims (int): 特征维度
        num_heads (int): 注意力头数
        feedforward_channels (int): FFN隐藏层通道数
        dropout (float): 注意力Dropout率
        ffn_dropout (float): FFN Dropout率
        operation_order (tuple[str]): 编码器中操作的顺序
        norm_cfg (dict): 规范化配置
    """
    def __init__(self, 
                 embed_dims,
                 num_heads, 
                 feedforward_channels,
                 dropout=0.1, 
                 ffn_dropout=0.1,
                 operation_order=('self_attn', 'norm', 'ffn', 'norm'),
                 norm_cfg=dict(type='LN')):
        super(TransformerEncoderLayer, self).__init__()
        
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.feedforward_channels = feedforward_channels
        self.dropout = dropout
        
        # 自注意力模块
        self.self_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dims)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(ffn_dropout),
            nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(ffn_dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dims)
        
        self.operation_order = operation_order
        assert set(operation_order) == {'self_attn', 'norm', 'ffn'}
    
    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                query_key_padding_mask=None,
                **kwargs):
        """Transformer编码器层的前向传播"""
        identity = query
        
        # 自注意力
        query = self.norm1(query)
        q = k = v = query
        if query_pos is not None:
            q = q + query_pos
            k = k + (key_pos if key_pos is not None else query_pos)
            
        attn_out = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            v.transpose(0, 1),
            key_padding_mask=query_key_padding_mask)[0].transpose(0, 1)
            
        query = identity + attn_out
        
        # 前馈网络
        identity = query
        query = self.norm2(query)
        query = identity + self.ffn(query)
        
        return query