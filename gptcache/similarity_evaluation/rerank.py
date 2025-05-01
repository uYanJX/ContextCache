from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import numpy as np
# Import the DialogueSimilarityModel
import math
import torch.nn.functional as F

from gptcache.similarity_evaluation import SimilarityEvaluation
from gptcache.utils import import_torch

import_torch()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        if x.dim() == 3:  # [batch, seq_len, dim]
            return x + self.pe[:x.size(1), :]
        elif x.dim() == 4:  # [batch, n, seq_len, dim]
            return x + self.pe[:x.size(2), :].unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")


class DialogueSimilarityModel(nn.Module):
    def __init__(self, 
                 embed_dim=768, 
                 num_heads=4, 
                 num_layers=2,
                 dropout_rate=0.1, 
                 temperature=0.07, 
                 similarity_threshold=0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.similarity_threshold = similarity_threshold
        
        # 核心编码组件
        self.scale = embed_dim ** 0.5
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        # 使用 TransformerEncoder 替代手动构建的多头注意力层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=1*embed_dim,  # 使用自定义的前馈网络维度
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim)
        )
        
        # 增强的动态池化层
        self.dynamic_pooling = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.LayerNorm(embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),  # 添加dropout
            nn.Linear(embed_dim // 4, 1),
            nn.Softmax(dim=1)
        )
        
        # 轻量级无匹配分类器
        self.no_match_classifier = nn.Sequential(
            nn.Linear(4, 16),  # 扩大隐藏层
            nn.LayerNorm(16),  # 添加规范化层
            nn.GELU(),  # 使用GELU替代ReLU
            nn.Dropout(dropout_rate),  # 添加dropout
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化模型参数，使用Xavier初始化"""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _reshape_for_batch_processing(self, x, masks=None):
        b, n, seq_len, dim = x.shape
        x_reshaped = x.view(b * n, seq_len, dim)
        
        if masks is not None:
            masks_reshaped = masks.view(b * n, seq_len)
            return x_reshaped, masks_reshaped
        
        return x_reshaped, None

    def encode_dialogues(self, x, mask=None):
        """编码对话序列为上下文表示"""
        # 应用位置编码和缩放
        x = x * self.scale
        x = self.pos_encoding(x)
        
        # 使用Transformer编码器
        if mask is not None:
            # 注意：TransformerEncoder中的key_padding_mask要求True表示被忽略的位置
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)
        
        # 动态池化获取全局表示
        attn_weights = self.dynamic_pooling(x)
        global_repr = torch.sum(attn_weights * x, dim=1)
        
        return global_repr
        
    def forward(self, dialogues, masks=None):
        """处理一批对话并计算相似度"""
        b, n, seq_len, dim = dialogues.shape
        assert n >= 2, "需要至少一个查询和一个候选项"
        
        # 重塑张量进行批处理
        dialogues_reshaped, masks_reshaped = self._reshape_for_batch_processing(dialogues, masks)
        
        # 编码所有对话
        global_reprs = self.encode_dialogues(dialogues_reshaped, masks_reshaped)
        global_reprs = global_reprs.view(b, n, dim)
        
        # 分离查询和候选项
        query_repr = global_reprs[:, 0]  # [b, dim]
        candidate_reprs = global_reprs[:, 1:]  # [b, n-1, dim]
        
        # 计算余弦相似度
        query_norm = F.normalize(query_repr, p=2, dim=1).unsqueeze(1)  # [b, 1, dim]
        candidate_norm = F.normalize(candidate_reprs, p=2, dim=2)  # [b, n-1, dim]
        cosine_similarities = torch.bmm(query_norm, candidate_norm.transpose(1, 2)).squeeze(1)  # [b, n-1]
        cosine_similarities = cosine_similarities 
        
        # 计算相似度统计特征
        max_sim, _ = torch.max(cosine_similarities, dim=1)  # [b]
        mean_sim = torch.mean(cosine_similarities, dim=1)  # [b]
        sim_std = torch.std(cosine_similarities, dim=1)  # [b]
        
        # 计算最高与次高相似度的差距
        if cosine_similarities.size(1) > 1:
            sorted_sims, _ = torch.sort(cosine_similarities, dim=1, descending=True)
            sim_gap = sorted_sims[:, 0] - sorted_sims[:, 1]  # [b]
        else:
            sim_gap = torch.zeros_like(max_sim)  # 如果只有一个候选项
        
        # 构建紧凑的特征向量
        relation_features = torch.stack([
            max_sim,       # 最高相似度
            mean_sim,      # 平均相似度
            sim_std,       # 相似度标准差
            sim_gap        # 最高与次高相似度差距
        ], dim=1)  # [b, 4]
        
        # 预测无匹配概率
        no_match_scores = self.no_match_classifier(relation_features).squeeze(-1)  # [b]
        
        # 创建包含无匹配得分的相似度向量
        similarities = torch.cat([no_match_scores.unsqueeze(1), cosine_similarities], dim=1)  # [b, n]
        similarities = F.softmax(similarities, dim=1)
        
        return similarities
    
    def compute_loss(self, similarities, labels):
        """计算损失，同时考虑匹配和无匹配情况"""
        batch_size = similarities.size(0)
        
        # 创建目标分布
        target = torch.zeros_like(similarities)
        
        # 设置目标值
        for i in range(batch_size):
            if labels[i] == -1:
                # 无匹配情况
                target[i, 0] = 1.0
            else:
                # 匹配情况（加1是因为索引0保留给"无匹配"位置）
                target[i, labels[i] + 1] = 1.0
        
        # 应用温度缩放
        logits = similarities / self.temperature
        
        # 计算交叉熵损失
        log_probs = F.log_softmax(logits, dim=1)
        loss = -torch.sum(target * log_probs) / batch_size
        
        return loss


class DialogueMatchEvaluation(SimilarityEvaluation):
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
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the model
        self.model = DialogueSimilarityModel(
            embed_dim=embed_dim,
        )
        model_path = "/data/home/Jianxin/MyProject/ContextCache/results/exp_20250316_010836/best_model.pth"
        # Load pre-trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()


    def evaluation(
        self, context_embs, **_
    ) -> float:
        """Evaluate the similarity score between dialogues.

        :param src_dict: the query dictionary to evaluate with cache.
        :type src_dict: Dict
        :param cache_dict: the cache dictionary.
        :type cache_dict: Dict

        :return: evaluation score.
        """
        num_to_fill = self.topk - len(context_embs)+1
        if num_to_fill > 0:
            for _ in range(num_to_fill):
                context_embs.append(np.array([]))  

        masks = []
        for dialogue in context_embs:
            mask = np.zeros(self.max_seq_length, dtype=bool)
            if len(dialogue) < self.max_seq_length:
                mask[len(dialogue):] = True
            masks.append(mask)  
                
        process_embs = [self._pad_sequence(d) for d in context_embs]
        dialogues = torch.tensor(np.array(process_embs), dtype=torch.float32).unsqueeze(0).to(self.device)
        masks = torch.tensor(np.array(masks), dtype=torch.bool).unsqueeze(0).to(self.device)
    
        
        # Run model inference
        with torch.no_grad():
            similarities = self.model(dialogues, masks)
            pre = torch.argmax(similarities, dim=1)
            # Get the similarity score (excluding no-match position)
        print(similarities,pre)
        return pre.item()
        

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
    