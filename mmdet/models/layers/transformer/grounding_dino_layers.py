# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import ModuleList
from torch import Tensor
import pdb
from mmdet.models.utils.vlfuse_helper import SingleScaleBiAttentionBlock
from mmdet.utils import ConfigType, OptConfigType
from .deformable_detr_layers import (DeformableDetrTransformerDecoderLayer,
                                     DeformableDetrTransformerEncoder,
                                     DeformableDetrTransformerEncoderLayer)
from .detr_layers import DetrTransformerEncoderLayer
from .dino_layers import DinoTransformerDecoder
from .utils import MLP, get_text_sine_pos_embed
import torch.nn.functional as F
try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None

def debug_tensor(name, tensor):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"Warning: {name} contains NaN or Inf!")


class ImageGuidedTextWeighting(nn.Module):
    def __init__(self, embed_dims, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads

        self.img_pool = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims)
        )

        self.attention = nn.MultiheadAttention(
            embed_dims, 
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dims)
    def forward(self, img_feat, text_feat, text_mask=None):
        # 1. 提取图像全局表示
        img_global = torch.mean(img_feat, dim=1, keepdim=True)  # [bs, 1, embed_dims]
        img_global = self.img_pool(img_global)
        debug_tensor("img_global", img_global)
        # 2. 图像全局特征引导文本注意力

        # 这里text_mask全为True,导致weighted_text里面的值全为NAN
        weighted_text, _ = self.attention(
            query=img_global,
            key=text_feat,
            value=text_feat,
            key_padding_mask=text_mask if text_mask is not None else None
        )
        # 3. 残差连接和归一化
        weighted_text = self.norm(weighted_text + img_global)
        # 4. 计算注意力权重
        attention_weights = torch.bmm(text_feat, weighted_text.transpose(1, 2))
        attention_weights = torch.softmax(attention_weights, dim=1)
        # 5. 应用权重
        weighted_text_feat = text_feat * attention_weights
        return weighted_text_feat



# class FeatureAdaptLayer(nn.Module):
#     """
#     A shared adaptation layer for aligning text and image features before cross-attention.
#     """
#     def __init__(self, embed_dims, ffn_ratio=4, dropout=0.1):
#         super().__init__()
#         # 共享适配层 - 使用相同参数处理两种模态
#         self.embed_dims = embed_dims
#         ffn_dims = int(embed_dims * ffn_ratio)
        
#         # 线性投影层
#         self.linear1 = nn.Linear(embed_dims, ffn_dims)
#         self.linear2 = nn.Linear(ffn_dims, embed_dims)
        
#         # 归一化层
#         self.norm1 = nn.LayerNorm(embed_dims)
        
#         # 激活和dropout
#         self.activation = nn.GELU()
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
        
#     def forward(self, img_feat, text_feat):
#         """
#         对图像和文本特征进行适配对齐处理
        
#         Args:
#             img_feat (Tensor): 图像特征 [bs, num_img_tokens, embed_dims]
#             text_feat (Tensor): 文本特征 [bs, num_text_tokens, embed_dims]
            
#         Returns:
#             Tuple[Tensor, Tensor]: 处理后的图像和文本特征
#         """
#         # 处理图像特征
#         img_adapted = img_feat
#         img_adapted = self.norm1(img_adapted)
#         img_adapted = self.linear1(img_adapted)
#         img_adapted = self.activation(img_adapted)
#         img_adapted = self.dropout1(img_adapted)
#         img_adapted = self.linear2(img_adapted)
#         img_adapted = self.dropout2(img_adapted)
#         img_adapted = img_feat + img_adapted
        
#         # 处理文本特征
#         text_adapted = text_feat
#         text_adapted = self.norm1(text_adapted)  # 共享参数
#         text_adapted = self.linear1(text_adapted)
#         text_adapted = self.activation(text_adapted)
#         text_adapted = self.dropout1(text_adapted)
#         text_adapted = self.linear2(text_adapted)
#         text_adapted = self.dropout2(text_adapted)
#         text_adapted = text_feat + text_adapted      
#         return img_adapted, text_adapted

# class FeatureAdaptLayer(nn.Module):
#     """共享的适应层，用于文本和图像特征的共同适应"""
    
#     def __init__(self, embed_dims, ffn_dims, dropout=0.1):
#         super().__init__()
#         # 共享特征投影
#         self.shared_proj = nn.Linear(embed_dims, embed_dims)
#         # 特征适应FFN
#         self.shared_ffn = FFN(
#             embed_dims=embed_dims,
#             feedforward_channels=ffn_dims,
#             num_fcs=2,
#             act_cfg=dict(type='ReLU', inplace=True),
#             dropout=dropout)
#         self.norm1 = nn.LayerNorm(embed_dims)
#         self.norm2 = nn.LayerNorm(embed_dims)
        
#     def forward(self, visual_feat, text_feat):
#         """前向传播函数
        
#         Args:
#             visual_feat (Tensor): 图像特征，形状为[bs, num_tokens, embed_dims]
#             text_feat (Tensor): 文本特征，形状为[bs, text_len, embed_dims]
            
#         Returns:
#             Tuple[Tensor, Tensor]: 适应后的图像和文本特征
#         """
#         # 拼接特征以进行共享处理
#         bs, vis_len, dim = visual_feat.shape
#         text_len = text_feat.shape[1]
        
#         # 将特征拼接在一起进行共享处理
#         combined_feat = torch.cat([visual_feat, text_feat], dim=1)  # [bs, vis_len+text_len, dim]
        
#         # 共享特征投影
#         combined_feat = self.norm1(combined_feat + self.shared_proj(combined_feat))
        
#         # 共享FFN处理
#         combined_feat = self.norm2(combined_feat + self.shared_ffn(combined_feat))
        
#         # 分离特征
#         adapted_visual_feat = combined_feat[:, :vis_len]
#         adapted_text_feat = combined_feat[:, vis_len:]
        
#         return adapted_visual_feat, adapted_text_feat


class FeatureAdaptLayer(nn.Module):
    def __init__(self, embed_dims, bottleneck_dim):
        """
        初始化 Adapt 层
        :param embed_dims: 输入特征的维度（图像和文本特征维度需一致）
        :param bottleneck_dim: 瓶颈层的维度（通常远小于 embed_dims）
        """
        super(FeatureAdaptLayer, self).__init__()
        self.embed_dims = embed_dims
        self.bottleneck_dim = bottleneck_dim 
        
        # 图像特征的下投影（带门控机制）
        self.W_d_img = nn.Linear(embed_dims, bottleneck_dim)
        self.W_g_img = nn.Linear(embed_dims, bottleneck_dim)
        
        # 文本特征的下投影（带门控机制）
        self.W_d_text = nn.Linear(embed_dims, bottleneck_dim)
        self.W_g_text = nn.Linear(embed_dims, bottleneck_dim)
        
        # 共享的上投影
        self.W_u = nn.Linear(bottleneck_dim, embed_dims)
    
    def forward(self, img_feat, text_feat):
        """
        前向传播
        :param img_feat: 图像特征，形状为 (batch_size, seq_len_img, embed_dims)
        :param text_feat: 文本特征，形状为 (batch_size, seq_len_text, embed_dims)
        :return: 融合后的图像和文本特征
        """
        # 图像特征的下投影
        z_img = F.silu(self.W_d_img(img_feat)) * self.W_g_img(img_feat)
        # 文本特征的下投影
        z_text = F.silu(self.W_d_text(text_feat)) * self.W_g_text(text_feat)
        
        # 拼接下投影特征
        z_combined = torch.cat([z_img, z_text], dim=1)
        
        # 上投影
        adapt_output = self.W_u(z_combined)
        
        # 分割回图像和文本特征
        img_adapt = adapt_output[:, :img_feat.size(1), :]
        text_adapt = adapt_output[:, img_feat.size(1):, :]
        
        # 残差连接
        img_feat = img_feat + img_adapt
        text_feat = text_feat + text_adapt
        
        return img_feat, text_feat



class GroundingDinoTransformerDecoderLayer(DeformableDetrTransformerDecoderLayer):

    def __init__(self,
                 cross_attn_text_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 **kwargs) -> None:
        """Decoder layer of Deformable DETR."""
        self.cross_attn_text_cfg = cross_attn_text_cfg
        if 'batch_first' not in self.cross_attn_text_cfg:
            self.cross_attn_text_cfg['batch_first'] = True
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn_text = MultiheadAttention(**self.cross_attn_text_cfg)
        self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(4)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                **kwargs) -> Tensor:
        """Implements decoder layer in Grounding DINO transformer.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.
            memory_text (Tensor): Memory text. It has shape (bs, len_text,
                text_embed_dims).
            text_attention_mask (Tensor): Text token mask. It has shape (bs,
                len_text).

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        # self attention
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs)
        query = self.norms[0](query)
        # cross attention between query and text
        query = self.cross_attn_text(
            query=query,
            query_pos=query_pos,
            key=memory_text,
            value=memory_text,
            key_padding_mask=text_attention_mask)
        query = self.norms[1](query)
        # cross attention between query and image
        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs)
        query = self.norms[2](query)
        query = self.ffn(query)
        query = self.norms[3](query)

        return query


class GroundingDinoTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType, **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = ModuleList([
            SingleScaleBiAttentionBlock(**self.fusion_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims

        # self.feature_adapt_layer = FeatureAdaptLayer(self.embed_dims)


        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
        """Forward function of Transformer encoder.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, has shape
                (bs, num_queries, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(spatial_shapes, valid_ratios, device=query.device)


        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)




        # main process
        for layer_id, layer in enumerate(self.layers):



            if self.fusion_layers:
                output, memory_text = self.fusion_layers[layer_id](
                    visual_feature=output,
                    lang_feature=memory_text,
                    attention_mask_v=key_padding_mask,
                    attention_mask_l=text_attention_mask,
                )

            # output, memory_text = self.feature_adapt_layer(output, memory_text)

            if self.text_layers:
                text_num_heads = self.text_layers[layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )


            
                
            output = layer(
                query=output,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask)
            
        return output, memory_text

    
# class GroundingDinoTransformerEncoder(DeformableDetrTransformerEncoder):

#     def __init__(self, text_layer_cfg: ConfigType,
#                  fusion_layer_cfg: ConfigType,
#                   shared_adapt_cfg: ConfigType = dict(
#                      embed_dims=256, bottleneck_dim=512),
#                       **kwargs) -> None:
        

#         self.text_layer_cfg = text_layer_cfg
#         self.fusion_layer_cfg = fusion_layer_cfg
#         self.shared_adapt_cfg = shared_adapt_cfg
#         super().__init__(**kwargs)

#     def _init_layers(self) -> None:
#         """Initialize encoder layers."""
#         self.layers = ModuleList([
#             DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
#             for _ in range(self.num_layers)
#         ])
        
#         self.text_layers = ModuleList([
#             DetrTransformerEncoderLayer(**self.text_layer_cfg)
#             for _ in range(self.num_layers)
#         ])
#         self.fusion_layers = ModuleList([
#             SingleScaleBiAttentionBlock(**self.fusion_layer_cfg)
#             for _ in range(self.num_layers)
#         ])
#         self.embed_dims = self.layers[0].embed_dims

#        # 添加共享适应层
#         self.shared_adapt_layers = ModuleList([
#             FeatureAdaptLayer(**self.shared_adapt_cfg)
#             for _ in range(self.num_layers)
#         ])

#         if self.num_cp > 0:
#             if checkpoint_wrapper is None:
#                 raise NotImplementedError(
#                     'If you want to reduce GPU memory usage, \
#                     please install fairscale by executing the \
#                     following command: pip install fairscale.')
#             for i in range(self.num_cp):
#                 self.layers[i] = checkpoint_wrapper(self.layers[i])
#                 self.fusion_layers[i] = checkpoint_wrapper(
#                     self.fusion_layers[i])

#     def forward(self,
#                 query: Tensor,
#                 query_pos: Tensor,
#                 key_padding_mask: Tensor,
#                 spatial_shapes: Tensor,
#                 level_start_index: Tensor,
#                 valid_ratios: Tensor,
#                 memory_text: Tensor = None,
#                 text_attention_mask: Tensor = None,
#                 pos_text: Tensor = None,
#                 text_self_attention_masks: Tensor = None,
#                 position_ids: Tensor = None):
#         output = query
#         reference_points = self.get_encoder_reference_points(spatial_shapes, valid_ratios, device=query.device)


#         if self.text_layers:
#             # generate pos_text
#             bs, n_text, _ = memory_text.shape
#             if pos_text is None and position_ids is None:
#                 pos_text = (
#                     torch.arange(n_text,
#                                  device=memory_text.device).float().unsqueeze(
#                                      0).unsqueeze(-1).repeat(bs, 1, 1))
#                 pos_text = get_text_sine_pos_embed(
#                     pos_text, num_pos_feats=256, exchange_xy=False)
#             if position_ids is not None:
#                 pos_text = get_text_sine_pos_embed(
#                     position_ids[..., None],
#                     num_pos_feats=256,
#                     exchange_xy=False)




#         # main process
#         for layer_id, layer in enumerate(self.layers):



#             # 使用文本自注意力层更新文本特征
#             if self.text_layers:
#                 text_num_heads = self.text_layers[layer_id].self_attn_cfg.num_heads
#                 memory_text = self.text_layers[layer_id](
#                     query=memory_text,
#                     query_pos=(pos_text if pos_text is not None else None),
#                     attn_mask=~text_self_attention_masks.repeat(
#                         text_num_heads, 1, 1),  # note we use ~ for mask here
#                     key_padding_mask=None,
#                 )

#             # 使用共享适应层处理文本和图像特征
#             if self.shared_adapt_layers:
#                 output, memory_text = self.shared_adapt_layers[layer_id](
#                     img_feat=output, 
#                     text_feat=memory_text
#                 )


#             # 使用双向注意力进行跨模态融合
#             if self.fusion_layers:
#                 output, memory_text = self.fusion_layers[layer_id](
#                     visual_feature=output,
#                     lang_feature=memory_text,
#                     attention_mask_v=key_padding_mask,
#                     attention_mask_l=text_attention_mask,
#                 )


#             #  使用图像特定层更新图像特征   
#             output = layer(
#                 query=output,
#                 query_pos=query_pos,
#                 reference_points=reference_points,
#                 spatial_shapes=spatial_shapes,
#                 level_start_index=level_start_index,
#                 key_padding_mask=key_padding_mask)
            
#         return output, memory_text
    


class GroundingDinoTransformerDecoder(DinoTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            GroundingDinoTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)
