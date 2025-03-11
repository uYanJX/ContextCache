# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# ################################################### 
# # new idea
# class PositionalEncoding2(nn.Module):
#     def __init__(self, d_model, max_len=50):
#         super().__init__()
#         # Create positional encoding matrix
#         position = torch.arange(0, max_len).unsqueeze(1).float()
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        
#         pe = torch.zeros(max_len, d_model)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)
    
#     def forward(self, x):
#         # Add positional encoding to input
#         # Adjust for different input shapes
#         if x.dim() == 3:  # [batch, seq_len, dim]
#             return x + self.pe[:x.size(1), :]
#         elif x.dim() == 4:  # [batch, n, seq_len, dim]
#             return x + self.pe[:x.size(2), :].unsqueeze(0).unsqueeze(0)
#         else:
#             raise ValueError(f"Unexpected input shape: {x.shape}")

# class BatchDialogueSimilarityModel(nn.Module):
#     def __init__(self, embed_dim, num_heads, dropout_rate=0.1, num_layers=2, temperature=0.07):
#         super(BatchDialogueSimilarityModel, self).__init__()
#         self.embed_dim = embed_dim
#         self.temperature = temperature  # Temperature parameter for InfoNCE loss
        
#         # Multi-head attention layers
#         self.mha_layers = nn.ModuleList([
#             nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout_rate)
#             for _ in range(num_layers)
#         ])
#         self.norm_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        
#         self.norm_final = nn.LayerNorm(embed_dim)
#         self.scale = embed_dim ** 0.5
#         self.pos_encoding = PositionalEncoding2(embed_dim)
        
#         # Feedforward networks
#         self.fc = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.Dropout(dropout_rate)
#         )
        
#         # Dynamic attention pooling
#         self.dynamic_pooling = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.Tanh(),
#             nn.Dropout(dropout_rate), 
#             nn.Linear(embed_dim, 1),
#             nn.Softmax(dim=1)
#         )
        
#         # Global representation enhancement
#         self.global_pool = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(embed_dim // 2, embed_dim)
#         )
        
#         # Cross-attention for comparing query with candidates
#         self.cross_attention = nn.MultiheadAttention(
#             embed_dim=embed_dim, 
#             num_heads=num_heads, 
#             batch_first=True, 
#             dropout=dropout_rate
#         )

#     def _reshape_for_batch_processing(self, x, masks=None):
#         """
#         Reshape input tensors for batch processing
#         [b, n, seq_len, dim] -> [b*n, seq_len, dim]
#         """
#         b, n, seq_len, dim = x.shape
#         x_reshaped = x.view(b * n, seq_len, dim)
        
#         if masks is not None:
#             masks_reshaped = masks.view(b * n, seq_len)
#             return x_reshaped, masks_reshaped
        
#         return x_reshaped, None

#     def encode_dialogue_batch(self, x, mask=None):
#         """
#         Encode a batch of dialogue sequences into contextualized representations
        
#         Args:
#             x: Tensor of shape [b*n, seq_len, dim]
#             mask: Tensor of shape [b*n, seq_len] or None
            
#         Returns:
#             global_repr: Tensor of shape [b*n, dim]
#             contextualized: Tensor of shape [b*n, seq_len, dim]
#         """
#         x = F.dropout(x, p=0.1, training=self.training)
#         x = x * self.scale
#         x = self.pos_encoding(x)
        
#         # Apply multi-head attention layers
#         for layer, norm in zip(self.mha_layers, self.norm_layers):
#             attn_output, _ = layer(x, x, x, key_padding_mask=mask)
#             attn_output = F.dropout(attn_output, p=0.1, training=self.training) 
#             x = norm(x + attn_output)  
        
#         # Apply feedforward network
#         fc_output = self.fc(x)
#         x = self.norm_final(x + fc_output)
        
#         # Dynamic pooling to get weights for each token
#         attn_weights = self.dynamic_pooling(x)
        
#         # Weighted sum to get global representation
#         global_repr = torch.sum(attn_weights * x, dim=1)
#         global_repr = self.global_pool(global_repr)
        
#         return global_repr, x
        
#     def forward(self, dialogues, masks=None):
#         """
#         Process a batch of dialogues and compute similarities using InfoNCE loss framework
        
#         Args:
#             dialogues: Tensor of shape [b, n, seq_len, dim]
#                 where n >= 2, with the first item being the query
#                 and the rest being the candidates
#             masks: Tensor of shape [b, n, seq_len] or None
                
#         Returns:
#             similarities: Tensor of shape [b, n-1]
#             loss: InfoNCE loss
#         """
#         b, n, seq_len, dim = dialogues.shape
#         assert n >= 2, "Need at least one query and one candidate per batch"
        
#         # Reshape for batch processing
#         dialogues_reshaped, masks_reshaped = self._reshape_for_batch_processing(dialogues, masks)
        
#         # Encode all dialogues at once
#         global_reprs, contextualized = self.encode_dialogue_batch(dialogues_reshaped, masks_reshaped)
        
#         # Reshape back to [b, n, dim] for global representations
#         global_reprs = global_reprs.view(b, n, dim)
        
#         # Split query and candidates
#         query_repr = global_reprs[:, 0]  # [b, dim]
#         candidate_reprs = global_reprs[:, 1:]  # [b, n-1, dim]
        
#         # Calculate similarities
#         query_norm = F.normalize(query_repr, p=2, dim=1).unsqueeze(1)  # [b, 1, dim]
#         candidate_norm = F.normalize(candidate_reprs, p=2, dim=2)  # [b, n-1, dim]
        
#         # Compute similarity matrix: [b, 1, dim] × [b, dim, n-1] -> [b, 1, n-1]
#         similarities = torch.bmm(query_norm, candidate_norm.transpose(1, 2)).squeeze(1)  # [b, n-1]
        
#         return similarities
    
#     def compute_similarity(self, query_dialogues, candidate_dialogues, query_masks=None, candidate_masks=None):
#         """
#         Compute similarity between query and candidate dialogues (for inference)
        
#         Args:
#             query_dialogues: Tensor of shape [b, 1, seq_len, dim]
#             candidate_dialogues: Tensor of shape [b, m, seq_len, dim]
#             query_masks: Tensor of shape [b, 1, seq_len] or None
#             candidate_masks: Tensor of shape [b, m, seq_len] or None
            
#         Returns:
#             similarities: Tensor of shape [b, m]
#         """
#         b, _, seq_len, dim = query_dialogues.shape
#         _, m, _, _ = candidate_dialogues.shape
        
#         # Combine query and candidates
#         combined_dialogues = torch.cat([query_dialogues, candidate_dialogues], dim=1)  # [b, m+1, seq_len, dim]
        
#         # Combine masks if they exist
#         combined_masks = None
#         if query_masks is not None and candidate_masks is not None:
#             combined_masks = torch.cat([query_masks, candidate_masks], dim=1)  # [b, m+1, seq_len]
        
#         # Get representations
#         combined_dialogues_reshaped, combined_masks_reshaped = self._reshape_for_batch_processing(
#             combined_dialogues, combined_masks)
        
#         global_reprs, _ = self.encode_dialogue_batch(combined_dialogues_reshaped, combined_masks_reshaped)
#         global_reprs = global_reprs.view(b, m+1, dim)
        
#         # Split query and candidates
#         query_repr = global_reprs[:, 0]  # [b, dim]
#         candidate_reprs = global_reprs[:, 1:]  # [b, m, dim]
        
#         # Calculate similarities
#         query_norm = F.normalize(query_repr, p=2, dim=1).unsqueeze(1)  # [b, 1, dim]
#         candidate_norm = F.normalize(candidate_reprs, p=2, dim=2)  # [b, m, dim]
        
#         similarities = torch.bmm(query_norm, candidate_norm.transpose(1, 2)).squeeze(1)  # [b, m]
        
#         return similarities

# # Example usage
# def train_example():
#     # Setup parameters
#     embed_dim = 768
#     num_heads = 8
#     batch_size = 4
#     n_samples = 5  # 1 query + 4 candidates per batch
#     dialogue_len = 10
    
#     # Create model
#     model = BatchDialogueSimilarityModel(embed_dim=embed_dim, num_heads=num_heads)
    
#     # Create optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
#     # Training loop example
#     for epoch in range(3):
#         # Create dummy input data: [b, n, seq_len, dim]
#         dialogues = torch.randn(batch_size, n_samples, dialogue_len, embed_dim)
        
#         # Create dummy masks (True = padding tokens)
#         masks = torch.zeros(batch_size, n_samples, dialogue_len, dtype=torch.bool)
#         for i in range(batch_size):
#             for j in range(n_samples):
#                 # Variable length padding
#                 padding_start = 7 + (i + j) % 3
#                 if padding_start < dialogue_len:
#                     masks[i, j, padding_start:] = True
        
#         # Forward pass
#         similarities, loss = model(dialogues, masks)
        
#         # Backward pass and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        
#         # Print similarity matrix for the first batch
#         if epoch == 2:
#             print(f"Similarities (batch 0): {similarities[0]}")
    
#     return model

# # Inference example
# def inference_example(model):
#     # Setup parameters
#     embed_dim = 768
#     batch_size = 2
#     dialogue_len = 10
#     n_candidates = 3
    
#     # Create dummy query data: [b, 1, seq_len, dim]
#     query_dialogues = torch.randn(batch_size, 1, dialogue_len, embed_dim)
    
#     # Create dummy candidate data: [b, m, seq_len, dim]
#     candidate_dialogues = torch.randn(batch_size, n_candidates, dialogue_len, embed_dim)
    
#     # Create masks
#     query_masks = torch.zeros(batch_size, 1, dialogue_len, dtype=torch.bool)
#     query_masks[:, :, 8:] = True  # Last 2 tokens are padding
    
#     candidate_masks = torch.zeros(batch_size, n_candidates, dialogue_len, dtype=torch.bool)
#     for i in range(batch_size):
#         for j in range(n_candidates):
#             candidate_masks[i, j, 7+j:] = True  # Variable padding
    
#     # Compute similarities
#     with torch.no_grad():
#         similarities = model.compute_similarity(
#             query_dialogues, candidate_dialogues, query_masks, candidate_masks
#         )
    
#     print(f"Inference similarities: {similarities}")
    
#     # Find most similar candidates for each query
#     most_similar_idx = torch.argmax(similarities, dim=1)
#     print(f"Most similar candidate indices: {most_similar_idx}")
    
#     return similarities, most_similar_idx

# if __name__ == "__main__":
#     model = train_example()
#     inference_example(model)


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
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
            dim_feedforward=embed_dim,
            dropout=dropout_rate,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim)
        )
        
        # 简化的动态池化层
        self.dynamic_pooling = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # 轻量级无匹配分类器
        self.no_match_classifier = nn.Sequential(
            nn.Linear(4, 8),  # 只使用4个关键统计特征
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

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
    
    @torch.no_grad()
    def compute_similarity(self, query_dialogues, candidate_dialogues, query_masks=None, candidate_masks=None):
        """计算查询与候选对话之间的相似度（用于推理）"""
        b, _, seq_len, dim = query_dialogues.shape
        _, m, _, _ = candidate_dialogues.shape
        
        # 合并查询和候选项
        combined_dialogues = torch.cat([query_dialogues, candidate_dialogues], dim=1)
        
        # 合并掩码
        combined_masks = None
        if query_masks is not None and candidate_masks is not None:
            combined_masks = torch.cat([query_masks, candidate_masks], dim=1)
        
        # 获取表示
        combined_dialogues_reshaped, combined_masks_reshaped = self._reshape_for_batch_processing(
            combined_dialogues, combined_masks)
        
        global_reprs = self.encode_dialogues(combined_dialogues_reshaped, combined_masks_reshaped)
        global_reprs = global_reprs.view(b, m+1, dim)
        
        # 分离查询和候选项
        query_repr = global_reprs[:, 0]  # [b, dim]
        candidate_reprs = global_reprs[:, 1:]  # [b, m, dim]
        
        # 计算余弦相似度
        query_norm = F.normalize(query_repr, p=2, dim=1).unsqueeze(1)  # [b, 1, dim]
        candidate_norm = F.normalize(candidate_reprs, p=2, dim=2)  # [b, m, dim]
        cosine_similarities = torch.bmm(query_norm, candidate_norm.transpose(1, 2)).squeeze(1)  # [b, m]
        
        # 计算相似度统计特征
        max_sim, _ = torch.max(cosine_similarities, dim=1)  # [b]
        mean_sim = torch.mean(cosine_similarities, dim=1)  # [b]
        sim_std = torch.std(cosine_similarities, dim=1)  # [b]
        
        # 计算相似度差异
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
        
        # 组合无匹配得分和候选项相似度
        similarities = torch.cat([no_match_scores.unsqueeze(1), cosine_similarities], dim=1)  # [b, m+1]
        
        # 使用多重标准确定是否有匹配
        max_sim_idx = torch.argmax(similarities, dim=1)  # [b]
        
        # 判断是否存在匹配
        match_decision = torch.zeros(b, dtype=torch.bool, device=similarities.device)
        
        for i in range(b):
            # 使用简化的判断逻辑
            if max_sim_idx[i] == 0 or max_sim[i] < self.similarity_threshold:
                match_decision[i] = False
            else:
                match_decision[i] = True
        
        return similarities, match_decision


# 训练示例
def train_example():
    torch.manual_seed(42)  # 设置随机种子以确保可重复性
    
    # 设置参数
    embed_dim = 768
    num_heads = 4
    batch_size = 4
    n_samples = 5  # 1 查询 + 4 候选项
    dialogue_len = 10
    
    # 创建模型
    model = DialogueSimilarityModel(
        embed_dim=embed_dim, 
        num_heads=num_heads,
        num_layers=2
    )
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # 训练循环示例
    for epoch in range(3):
        # 创建模拟输入数据
        dialogues = torch.randn(batch_size, n_samples, dialogue_len, embed_dim)
        
        # 创建掩码
        masks = torch.zeros(batch_size, n_samples, dialogue_len, dtype=torch.bool)
        for i in range(batch_size):
            for j in range(n_samples):
                padding_start = 7 + (i + j) % 3
                if padding_start < dialogue_len:
                    masks[i, j, padding_start:] = True
        
        # 创建标签
        labels = torch.tensor([1, -1, 2, -1], dtype=torch.long)
        
        # 前向传播
        similarities = model(dialogues, masks)
        
        # 计算损失
        loss = model.compute_loss(similarities, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    return model


if __name__ == "__main__":
    model = train_example()