from typing import Dict, List, Tuple, Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  

from gptcache.similarity_evaluation import SimilarityEvaluation
from gptcache.utils import import_torch

import_torch()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super(PositionalEncoding, self).__init__()
        # 创建一个 (max_len, d_model) 的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 增加batch维度
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将位置编码加到输入 x 中
        x = x + self.pe[:, :x.size(1), :]
        return x
    
class AttentionLayer(nn.Module):
    """单层注意力模块，保持与原模型相同的注意力计算逻辑"""
    def __init__(self, embed_dim, num_heads=8, dropout_rate=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "嵌入维度必须能被头数整除"
        
        # 多头注意力参数
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        
        # 投影层
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # 层归一化
        self.norm = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # 缩放因子
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query, key_value, mask=None):
        """
        Args:
            query: 查询向量 [batch_size, embed_dim]
            key_value: 键值序列 [batch_size, seq_len, embed_dim]
            mask: 掩码 [batch_size, seq_len]
        """
        batch_size = key_value.size(0)
        residual = query  # 用于残差连接
        
        # 维度变换 - 查询是单个向量
        q = self.query_proj(query).view(batch_size, 1, self.num_heads, self.head_dim)
        k = self.key_proj(key_value).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.value_proj(key_value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # 调整维度顺序
        q = q.permute(0, 2, 1, 3)  # [batch, heads, 1, head_dim]
        k = k.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim]
        v = v.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim]
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask, -1e9)
        
        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 应用注意力权重
        context = torch.matmul(attn_weights, v)
        
        # 重塑回原始维度
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, 1, self.embed_dim)
        output = self.output_proj(context).squeeze(1)
        
        # 残差连接和归一化
        output = self.norm(output + residual)
        
        return output, attn_weights

class QueryContextTransformerLayer(nn.Module):
    """单层查询上下文Transformer
    
    将查询与序列进行交互的单层处理单元
    """
    def __init__(self, embed_dim, num_heads, dropout_rate, ffn_dim):
        super().__init__()
        
        # 交叉注意力层
        self.cross_attention = AttentionLayer(embed_dim, num_heads, dropout_rate)
        
    def forward(self, query, context_vectors, mask=None):
        """单层Transformer处理
        
        Args:
            query: [batch_size, embed_dim] - 当前查询状态
            context_vectors: [batch_size, seq_len, embed_dim] - 上下文序列
            mask: [batch_size, seq_len] - 掩码 (True=填充)
            
        Returns:
            [batch_size, embed_dim] - 更新后的查询状态
        """
        context, _ = self.cross_attention(
            query,
            context_vectors,
            mask
        )
        return context  # [batch, dim]


class BatchOptimizedEncoder(nn.Module):
    """批量优化版全局查询上下文编码器
    
    特点:
    1. 支持批量处理所有样本（锚点、正样本、负样本）
    2. 单次前向传递处理所有数据
    3. 保留多层Transformer架构和其他功能特性
    4. 显著提高训练效率
    """
    def __init__(self, 
                 embed_dim, 
                 num_heads=8, 
                 num_layers=3,
                 dropout_rate=0.2, 
                 ffn_dim_multiplier=2,
                 normalize_output=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.normalize_output = normalize_output
            
        # 多层Transformer块
        self.layers = nn.ModuleList([
            QueryContextTransformerLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                ffn_dim=embed_dim * ffn_dim_multiplier
            )
            for _ in range(num_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
    def extract_last_valid_vector(self, sequence, mask=None):
        """提取序列中最后一个非填充的向量"""
        if mask is None:
            return sequence[:, -1, :]
        else:
            last_indices = (~mask).sum(dim=1) - 1
            batch_size = sequence.size(0)
            batch_indices = torch.arange(batch_size, device=sequence.device)
            return sequence[batch_indices, last_indices]

    def forward(self, vectors, mask=None):
        """单个样本的前向传递"""
        last_query = self.extract_last_valid_vector(vectors, mask)
        query_state = last_query
        for layer in self.layers:
            query_state = layer(query_state, vectors, mask)
            
        output = self.output_proj(query_state)
        
        if self.normalize_output:
            output = F.normalize(output, p=2, dim=-1)
            
        return output
    
    def batch_forward(self, batch_data, batch_mask):
        """批量处理所有样本的高效前向传递
        
        Args:
            batch_data: [batch_size, num_samples, seq_len, feature_dim] 
                        包含锚点、正样本和负样本的批次数据
            batch_mask: [batch_size, num_samples, seq_len] 对应的掩码
            
        Returns:
            tuple: (anchor_repr, pos_repr, neg_reprs)
                anchor_repr: [batch_size, embed_dim]
                pos_repr: [batch_size, embed_dim]
                neg_reprs: list of [batch_size, embed_dim]
        """
        batch_size, num_samples = batch_data.shape[0], batch_data.shape[1]
        # 1. 重塑数据以便批量处理
        # 将所有样本在批维度上合并: [batch_size*num_samples, seq_len, feature_dim]
        flat_data = batch_data.reshape(-1, *batch_data.shape[2:])
        flat_mask = batch_mask.reshape(-1, batch_mask.shape[-1])
        flat_data = F.dropout(flat_data, p=0.1, training=self.training)
        
        # 2. 单次前向传递处理所有样本
        all_outputs = self.forward(flat_data, flat_mask)  # [batch_size*num_samples, embed_dim]
        
        # 3. 重新整形回原始批次形状
        reshaped_outputs = all_outputs.reshape(batch_size, num_samples, -1)  # [batch_size, num_samples, embed_dim]
        
        # 4. 分离结果
        anchor_repr = reshaped_outputs[:, 0]  # 锚点是第一个样本
        other_reprs = reshaped_outputs[:, 1:]  # 其他样本
        
        return anchor_repr, other_reprs

class AttentionContext(nn.Module):
    """多层注意力实现的MLastVectorAttention"""
    def __init__(self, embed_dim, num_layers=2, num_heads=12, dropout_rate=0.1):
        super(AttentionContext, self).__init__()
        
        # 确保embed_dim可以被头数整除
        assert embed_dim % num_heads == 0, "嵌入维度必须能被头数整除"
        
        # 基础组件
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 多层注意力
        self.attention_layers = nn.ModuleList([
            AttentionLayer(embed_dim, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])
        
        # 最终的输出转换
        self.final_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
    
    def extract_last_valid_vector(self, sequence, mask=None):
        """提取序列中最后一个非填充的向量"""
        if mask is None:
            # 如果没有掩码，直接取最后一个向量
            return sequence[:, -1, :]
        else:
            # 获取每个序列中最后一个有效位置的索引
            last_indices = (~mask).sum(dim=1) - 1
            
            # 从每个序列中提取最后一个有效向量
            batch_size = sequence.size(0)
            batch_indices = torch.arange(batch_size, device=sequence.device)
            last_vectors = sequence[batch_indices, last_indices]
            
            return last_vectors  # [batch_size, embed_dim]
    
    def process_sequence(self, sequence, mask=None):
        """通过多层注意力处理序列"""
        batch_size = sequence.size(0)
        
        # 获取最后一个有效向量作为初始查询
        query = self.extract_last_valid_vector(sequence, mask)
        
        all_attn_weights = []
        
        # 逐层处理
        for layer in self.attention_layers:
            query, attn_weights = layer(query, sequence, mask)
            all_attn_weights.append(attn_weights)
        
        # 最终的非线性变换
        global_repr = self.final_proj(query)
        
        # 为了便于后续分析，返回最后一层的注意力权重
        last_attn_weights = all_attn_weights[-1].mean(dim=1).squeeze(1)  # [batch_size, seq_len]
        
        return global_repr, last_attn_weights
    
    def forward(self, a_vectors, b_vectors, mask_a=None, mask_b=None):
        """
        计算两段对话的全局表示
        
        Args:
            a_vectors: 第一段对话的向量表示 [batch_size, seq_len_a, embed_dim]
            b_vectors: 第二段对话的向量表示 [batch_size, seq_len_b, embed_dim]
            mask_a: 第一段对话的掩码 [batch_size, seq_len_a] (True表示需要掩盖)
            mask_b: 第二段对话的掩码 [batch_size, seq_len_b] (True表示需要掩盖)
        """
        # 检查输入维度
        assert a_vectors.dim() == 3 and b_vectors.dim() == 3, "输入向量必须是3维的"
        
        # 输入处理
        a = self.dropout(a_vectors)
        b = self.dropout(b_vectors)
        
        # 应用位置编码和归一化
        a = self.input_norm(self.pos_encoding(a))
        b = self.input_norm(self.pos_encoding(b))
        
        # 获取全局表示
        a_global_repr, a_weights = self.process_sequence(a, mask_a)
        b_global_repr, b_weights = self.process_sequence(b, mask_b)
        
        return a_global_repr, b_global_repr
    
class MLastVectorAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=12, dropout_rate=0.2):
        super(MLastVectorAttention, self).__init__()
        
        # 确保embed_dim可以被头数整除
        assert embed_dim % num_heads == 0, "嵌入维度必须能被头数整除"
        
        # 基础组件
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 多头注意力参数
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        
        # 用于多头注意力计算的线性投影层
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # 最终的输出转换
        self.final_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # 缩放因子
        self.scale = self.head_dim ** -0.5
    
    def extract_last_valid_vector(self, sequence, mask=None):
        """提取序列中最后一个非填充的向量"""
        if mask is None:
            # 如果没有掩码，直接取最后一个向量
            return sequence[:, -1, :]
        else:
            # 获取每个序列中最后一个有效位置的索引
            # (~mask)将True/False反转，使有效位置为True
            # sum(1)沿着时间维度求和，得到每个序列中有效位置的数量
            # -1 得到最后一个有效位置的索引
            last_indices = (~mask).sum(dim=1) - 1
            
            # 从每个序列中提取最后一个有效向量
            batch_size = sequence.size(0)
            batch_indices = torch.arange(batch_size, device=sequence.device)
            last_vectors = sequence[batch_indices, last_indices]
            
            return last_vectors  # [batch_size, embed_dim]
    
    def compute_multihead_self_attention(self, sequence, mask=None):
        """使用多头注意力机制计算序列的全局表示"""
        batch_size = sequence.size(0)
        
        # 获取最后一个有效向量作为查询
        last_vector = self.extract_last_valid_vector(sequence, mask)  # [batch_size, embed_dim]
        
        q = self.query_proj(last_vector).unsqueeze(1).view(batch_size, 1, self.num_heads, self.head_dim)
        # [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, num_heads, head_dim]
        k = self.key_proj(sequence).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.value_proj(sequence).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # 调整维度顺序以便计算注意力
        # [batch_size, 1, num_heads, head_dim] -> [batch_size, num_heads, 1, head_dim]
        q = q.permute(0, 2, 1, 3)
        # [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        # 计算注意力分数
        # [batch_size, num_heads, 1, head_dim] @ [batch_size, num_heads, head_dim, seq_len]
        # -> [batch_size, num_heads, 1, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # 应用掩码
        if mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            mask = mask.unsqueeze(1).unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask, -1e9)
        
        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, num_heads, 1, seq_len]
        
        # 应用注意力权重
        # [batch_size, num_heads, 1, seq_len] @ [batch_size, num_heads, seq_len, head_dim]
        # -> [batch_size, num_heads, 1, head_dim]
        context = torch.matmul(attn_weights, v)
        
        # 重塑回原始维度
        # [batch_size, num_heads, 1, head_dim] -> [batch_size, 1, num_heads, head_dim]
        context = context.permute(0, 2, 1, 3)
        # [batch_size, 1, num_heads, head_dim] -> [batch_size, 1, embed_dim]
        context = context.contiguous().view(batch_size, 1, self.embed_dim)
        
        # 应用输出投影
        context = self.output_proj(context).squeeze(1)  # [batch_size, embed_dim]
        
        # 最终的非线性变换
        global_repr = self.final_proj(context)
        
        # 为了便于后续分析，返回平均注意力权重
        avg_attn_weights = attn_weights.mean(dim=1).squeeze(1)  # [batch_size, seq_len]
        
        return global_repr, avg_attn_weights
    
    def forward(self, a_vectors, b_vectors, mask_a=None, mask_b=None):
        """
        计算两段对话的全局表示
        
        Args:
            a_vectors: 第一段对话的向量表示 [batch_size, seq_len_a, embed_dim]
            b_vectors: 第二段对话的向量表示 [batch_size, seq_len_b, embed_dim]
            mask_a: 第一段对话的掩码 [batch_size, seq_len_a] (True表示需要掩盖)
            mask_b: 第二段对话的掩码 [batch_size, seq_len_b] (True表示需要掩盖)
        """
        # 检查输入维度
        assert a_vectors.dim() == 3 and b_vectors.dim() == 3, "输入向量必须是3维的"
        
        # 输入处理
        a = self.dropout(a_vectors)
        b = self.dropout(b_vectors)
        
        # 应用位置编码和归一化
        a = self.norm(self.pos_encoding(a))
        b = self.norm(self.pos_encoding(b))
        
        # 获取全局表示
        a_global_repr, a_weights = self.compute_multihead_self_attention(a, mask_a)
        b_global_repr, b_weights = self.compute_multihead_self_attention(b, mask_b)
        
        return a_global_repr, b_global_repr

class CrossAttentionMHA(nn.Module):
    def __init__(self, embed_dim, num_heads=12, dropout_rate=0.2, num_layers=1):
        super(CrossAttentionMHA, self).__init__()
        
        # 自注意力层 - 捕捉序列内部依赖
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout_rate)
            for _ in range(num_layers)
        ])
        
        # 交叉注意力层 - 建立序列间关联
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout_rate)
            for _ in range(num_layers)
        ])
        
        # 各种归一化层
        self.self_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.cross_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # 前馈网络 - 处理交互后的特征
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 2 * embed_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(2 * embed_dim, embed_dim),
            ) for _ in range(num_layers)
        ])
        self.ffn_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        # 适应性池化 - 动态加权聚合
        self.pooling = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim//2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim//2, 1),
            nn.Softmax(dim=1)
        )
        
        self.dropout_rate = dropout_rate
        
    def encode_sequence(self, x, mask=None):
        """预处理序列：应用缩放和位置编码"""
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.pos_encoding(x)  
        return x
    
    def apply_mask(self, x, mask=None):
        """应用掩码，确保padding位置不影响结果"""
        if mask is not None:
            expanded_mask = (~mask).unsqueeze(-1)
            x = x.masked_fill(expanded_mask == 0, 0.0)
        return x
        
    def forward(self, x1, x2, mask1=None, mask2=None):
        """
        级联处理两组语义向量序列
        
        Args:
            x1: 第一组语义向量 [batch_size, seq_len1, embed_dim]
            x2: 第二组语义向量 [batch_size, seq_len2, embed_dim]
            mask1: 第一组向量的padding掩码
            mask2: 第二组向量的padding掩码
            
        Returns:
            similarity: 两组向量的相似度得分
            x1_global: 第一组向量的全局表示
            x2_global: 第二组向量的全局表示
        """
        
        # 初始预处理
        x1 = self.encode_sequence(x1)
        x2 = self.encode_sequence(x2)
        
        # # 保存原始输入，用于残差连接
        x1_residual = x1
        x2_residual = x2
        
        # 多层级联处理
        for i in range(len(self.self_attn_layers)):
            # ===== 第一层：自注意力 =====
            # 序列1自注意力
            self_attn1, _ = self.self_attn_layers[i](x1, x1, x1, key_padding_mask=mask1)
            self_attn1 = self.apply_mask(self_attn1, mask1)
            x1 = self.self_norm[i](x1 + self_attn1)
            
            # 序列2自注意力
            self_attn2, _ = self.self_attn_layers[i](x2, x2, x2, key_padding_mask=mask2)
            self_attn2 = self.apply_mask(self_attn2, mask2)
            x2 = self.self_norm[i](x2 + self_attn2)
            
            # ===== 第二层：交叉注意力 =====
            # 序列1关注序列2
            cross_attn1, attn_weights1 = self.cross_attn_layers[i](x1, x2, x2, key_padding_mask=mask2)
            cross_attn1 = self.apply_mask(cross_attn1, mask1)
            x1 = self.cross_norm[i](x1 + cross_attn1)
            
            # 序列2关注序列1
            cross_attn2, attn_weights2 = self.cross_attn_layers[i](x2, x1, x1, key_padding_mask=mask1)
            cross_attn2 = self.apply_mask(cross_attn2, mask2)
            x2 = self.cross_norm[i](x2 + cross_attn2)
            
            # ===== FFN层 =====
            x1_ffn = self.ffn[i](x1)
            x1_ffn = self.apply_mask(x1_ffn, mask1)
            x1 = self.ffn_norm[i](x1 + x1_ffn)
            
            x2_ffn = self.ffn[i](x2)
            x2_ffn = self.apply_mask(x2_ffn, mask2)
            x2 = self.ffn_norm[i](x2 + x2_ffn)
        
        # # 全局残差连接 - 连接到初始输入
        x1 = self.final_norm(x1 + x1_residual)
        x2 = self.final_norm(x2 + x2_residual)
        
        # 适应性池化 - 生成全局表示
        # 处理序列1
        attn_weights1 = self.pooling(x1)
        if mask1 is not None:
            attn_weights1 = self.apply_mask(attn_weights1, mask1)
            attn_sum1 = attn_weights1.sum(dim=1, keepdim=True) + 1e-9
            attn_weights1 = attn_weights1 / attn_sum1
        x1_global = torch.sum(attn_weights1 * x1, dim=1)
        
        # 处理序列2
        attn_weights2 = self.pooling(x2)
        if mask2 is not None:
            attn_weights2 = self.apply_mask(attn_weights2, mask2)
            attn_sum2 = attn_weights2.sum(dim=1, keepdim=True) + 1e-9
            attn_weights2 = attn_weights2 / attn_sum2
        x2_global = torch.sum(attn_weights2 * x2, dim=1)
        
        return x1_global, x2_global


class OptimizedMHA(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout_rate=0.15, num_layers=2, ff_dim_factor=4):
        super(OptimizedMHA, self).__init__()
        
        self.scale = embed_dim ** 0.5
        self.dropout_rate = dropout_rate
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        # Build transformer encoder-style layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, 
                                     batch_first=True, dropout=dropout_rate),
                nn.LayerNorm(embed_dim),
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * ff_dim_factor),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(embed_dim * ff_dim_factor, embed_dim),
                    nn.Dropout(dropout_rate)
                ),
                nn.LayerNorm(embed_dim)
            ]))
        
        # Dynamic attention pooling
        self.dynamic_pooling = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Final representation transformation
        self.global_pool = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        self._init_parameters()
    
    def _init_parameters(self):
        # Xavier initialization for linear layers
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_normal_(param)
    
    def forward(self, x, key_padding_mask=None):
        # Initial dropout and scaling
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = x * self.scale
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Process through transformer-style layers
        for mha, norm1, ff, norm2 in self.layers:
            # Multi-head attention block
            attn_output, _ = mha(x, x, x, key_padding_mask=key_padding_mask)
            attn_output = F.dropout(attn_output, p=self.dropout_rate, training=self.training)
            x = norm1(x + attn_output)  # Residual connection and normalization
            
            # Feed-forward block
            ff_output = ff(x)
            x = norm2(x + ff_output)  # Residual connection and normalization
        
        # Dynamic attention pooling to create weighted sum
        attn_weights = self.dynamic_pooling(x)
        
        # Create global representation
        global_repr = torch.sum(attn_weights * x, dim=1)
        global_repr = self.global_pool(global_repr)
        
        return global_repr

class OptimizedMHA5(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout_rate=0.1,num_layers=3):
        super(OptimizedMHA5, self).__init__()
        # self.linear = nn.Linear(embed_dim, embed_dim)
        self.mha_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout_rate)
            for _ in range(num_layers)
        ])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])

        self.norm2 = nn.LayerNorm(embed_dim)
        self.scale = embed_dim ** 0.5
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout_rate)  # 添加Dropout
        )
        
        self.dynamic_pooling = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate), 
            nn.Linear(embed_dim, 1),
            nn.Softmax(dim=1)
        )
        
        self.global_pool = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),  # 降低中间维度
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 添加Dropout
            nn.Linear(embed_dim, embed_dim)
        )

        
    def forward(self, x, key_padding_mask):
        x = F.dropout(x, p=0.1, training=self.training)
        x = x * self.scale
        x = self.pos_encoding(x)
        for layer, norm in zip(self.mha_layers, self.norm_layers):
            attn_output, _ = layer(x, x, x, key_padding_mask=key_padding_mask)
            attn_output = F.dropout(attn_output, p=0.1, training=self.training) 
            x = norm(x + attn_output)  
        fc_output = self.fc(x)
        x = self.norm2(x + fc_output)
        attn_weights = self.dynamic_pooling(x)
        global_repr = torch.sum(attn_weights * x, dim=1)
        global_repr = self.global_pool(global_repr)
        
        return global_repr

    
class ContextMatchEvaluation(SimilarityEvaluation):
    """Using DialogueSimilarityModel to evaluate dialogue pair similarity.

    This evaluator uses a transformer-based model to evaluate the similarity between dialogues.

    :param model_path: path to the pre-trained DialogueSimilarityModel. Required.
    :type model_path: str
    :param embed_dim: dimension of embeddings used in the model. Default is 768.
    :type embed_dim: int
    :param max_seq_len: maximum sequence length for dialogues. Default is 20.
    :type max_seq_len: int
    :param device: device to run the model on (e.g., 'cpu', 'cuda'). Default is None (auto-detect).
    :type device: str

    Example:
        .. code-block:: python

            from gptcache.similarity_evaluation import DialogueSimilarityEvaluation

            evaluation = DialogueSimilarityEvaluation(model_path="path/to/model.pt")
            score = evaluation.evaluation(
                {
                    "question": "How's the weather today?"
                },
                {
                    "question": "What's the weather like?"
                }
            )
    """

    def __init__(
        self, 
        embed_dim: int = 768, 
        max_seq_len: int = 10,
        device: str = None
    ):
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_len
        self.topk = 5
        
        # Set device
        self.device = device if device is not None else ("cuda:7" if torch.cuda.is_available() else "cpu")
        
        # Load the model
        self.model = BatchOptimizedEncoder(
            embed_dim=embed_dim,
        )
        ## neg hard
        # model_path = "/data/home/Jianxin/MyProject/ContextCache/results/exp_20250319_234817_1/best_model.pth"
        
        # ## no neg hard  
        # model_path = "/data/home/Jianxin/MyProject/ContextCache/results/exp_20250415_222257_7/best_model.pth"
        
        ## cross results/exp_20250417_210833_4/checkpoint_epoch_30.pth
        
        ## atten /data/home/Jianxin/MyProject/ContextCache/results/exp_20250424_100746_7/best_model_2.pth
        
        # model_path = "/data/home/Jianxin/MyProject/ContextCache/results/exp_20250425_151828_7/checkpoint_epoch_20.pth"
        
        model_path = "/data/home/Jianxin/MyProject/ContextCache/results/exp_20250425_234420_6/checkpoint_epoch_20.pth"

        # /data/home/Jianxin/MyProject/ContextCache/results/exp_20250425_234420_6/best_model_2.pth

        # Load pre-trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # # Load the model
        # self.model = OptimizedMHA(
        #     embed_dim=embed_dim,
        # )
        # model_path = "/data/home/Jianxin/MyProject/ContextCache/results/exp_20250413_011317_1/best_model.pth"
        # # Load pre-trained weights
        # checkpoint = torch.load(model_path, map_location=self.device)    
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()


    def evaluation(
        self, ori_emb, test_emb, **_
    ) -> float:
        """Evaluate the similarity score between dialogues.

        :param src_dict: the query dictionary to evaluate with cache.
        :type src_dict: Dict
        :param cache_dict: the cache dictionary.
        :type cache_dict: Dict

        :return: evaluation score.
        """


        mask_ori = np.zeros(self.max_seq_length, dtype=bool)
        if len(ori_emb) < self.max_seq_length:
            mask_ori[len(ori_emb):] = True
        
        mask_test = np.zeros(self.max_seq_length, dtype=bool)
        if len(test_emb) < self.max_seq_length:
            mask_test[len(test_emb):] = True
                
        ori_emb = self._pad_sequence(ori_emb)
        test_emb = self._pad_sequence(test_emb)
    
        cos = nn.CosineSimilarity(dim=1)
        
        # Run model inference
        with torch.no_grad():
            s1 = torch.tensor(ori_emb, dtype=torch.float32, device=self.device).unsqueeze(0)
            s2 = torch.tensor(test_emb, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_s1 = torch.tensor(mask_ori, dtype=torch.bool, device=self.device).unsqueeze(0)
            mask_s2 = torch.tensor(mask_test, dtype=torch.bool, device=self.device).unsqueeze(0)
            repr_s1 = self.model(s1, mask_s1)
            repr_s2 = self.model(s2, mask_s2)
            # repr_s1, repr_s2 = self.model(s1, s2, mask_s1, mask_s2)
            
            
            # Get the similarity score (excluding no-match position)
        repr_s1 = F.normalize(repr_s1, p=2, dim=1)
        repr_s2 = F.normalize(repr_s2, p=2, dim=1)
        
        score = (cos(repr_s1, repr_s2).item())
        return score
        
    # def evaluation(
    #     self, ori_emb, test_emb=None ,**_
    # ) -> float:
    #     """Evaluate the similarity score between dialogues.

    #     :param src_dict: the query dictionary to evaluate with cache.
    #     :type src_dict: Dict
    #     :param cache_dict: the cache dictionary.
    #     :type cache_dict: Dict

    #     :return: evaluation score.
    #     """


    #     mask_ori = np.zeros(self.max_seq_length, dtype=bool)
    #     if len(ori_emb) < self.max_seq_length:
    #         mask_ori[len(ori_emb):] = True
                
    #     ori_emb = self._pad_sequence(ori_emb)
    
    #     cos = nn.CosineSimilarity(dim=1)
        
    #     # Run model inference
    #     with torch.no_grad():
    #         s1 = torch.tensor(ori_emb, dtype=torch.float32, device=self.device).unsqueeze(0)
    #         mask_s1 = torch.tensor(mask_ori, dtype=torch.bool, device=self.device).unsqueeze(0)
    #         repr_s1 = self.model(s1, mask_s1)
    #         # repr_s1, repr_s2 = self.model(s1, s2, mask_s1, mask_s2)
            
            
    #         # Get the similarity score (excluding no-match position)
    #     if test_emb is not None:
    #         score = cos(repr_s1, test_emb).item()
    #     else:
    #         score = 1
    #     return score, repr_s1


    def range(self) -> Tuple[float, float]:
        """Range of similarity score.

        :return: minimum and maximum of similarity score.
        """
        return 0.0, 1.0
    
    
    def _pad_sequence(self, sequence):
        """对序列进行填充或截断"""
        if len(sequence) == 0:
            # 处理空序列的情况
            embed_dim = self.embed_dim
            return np.zeros((self.max_seq_length, embed_dim))
        
        if len(sequence) > self.max_seq_length:
            return sequence[:self.max_seq_length]
        
        padded_shape = (self.max_seq_length, sequence.shape[1])
        padded = np.zeros(padded_shape)
        padded[:len(sequence)] = sequence
        return padded
    