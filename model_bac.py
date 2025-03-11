# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=10):
#         super(PositionalEncoding, self).__init__()
#         # 创建一个 (max_len, d_model) 的位置编码矩阵
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)  # 增加batch维度
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         # 将位置编码加到输入 x 中
#         x = x + self.pe[:, :x.size(1), :]
#         return x

     
# class OptimizedMHA(nn.Module):
#     def __init__(self, input_dim, embed_dim, num_heads):
#         super(OptimizedMHA, self).__init__()
#         self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.scale = embed_dim ** 0.5
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.pos_encoding = PositionalEncoding(embed_dim)
#         self.fc = nn.Linear(embed_dim, embed_dim)
#         self.global_pool = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim // 2),
#             nn.ReLU(),
#             nn.Linear(embed_dim // 2, embed_dim)
#         )

#     def forward(self, x, key_padding_mask):
#         # Multi-Head Attention with padding mask
#         x = x * self.scale
#         x = self.pos_encoding(x)
#         attn_output, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
#         # Residual connection and normalization
#         x = self.norm1(x + attn_output)
#         # Fully connected transformation
#         fc_output = self.fc(x)
#         # Residual connection and normalization
#         x = self.norm2(x + fc_output)
#         # Global aggregation with masking (ensuring padded positions are excluded)
#         mask = ~key_padding_mask.unsqueeze(-1)
#         masked_output = x * mask
#         # Weighted global representation
#         global_repr = (masked_output.sum(dim=1) / mask.sum(dim=1))
#         global_repr = self.global_pool(global_repr)
#         return global_repr 

# # best 0.83~ 
# class OptimizedMHA2(nn.Module):
#     def __init__(self,  embed_dim, num_heads, dropout_rate=0.1):
#         super(OptimizedMHA2, self).__init__()
#         self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True,dropout=0.1)
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.scale = embed_dim ** 0.5
#         self.pos_encoding = PositionalEncoding(embed_dim)
#         self.fc = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.Dropout(dropout_rate)  # 添加Dropout
#         )
#         self.global_pool = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim // 2),  # 降低中间维度
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),  # 添加Dropout
#             nn.Linear(embed_dim // 2, embed_dim)
#         )
#         self.attention_weights = nn.Sequential(
#             nn.Linear(embed_dim, 1),
#             nn.Softmax(dim=1)
#         )

#     def forward(self, x, key_padding_mask):
#         x = F.dropout(x, p=0.1, training=self.training)
#         x = x * self.scale
#         x = self.pos_encoding(x)
#         attn_output, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
#         attn_output = F.dropout(attn_output, p=0.1, training=self.training) 
#         x = self.norm1(x + attn_output)
#         fc_output = self.fc(x)
#         x = self.norm2(x + fc_output)
#         attn_weights = self.attention_weights(x)
#         global_repr = torch.sum(attn_weights * x, dim=1)
#         global_repr = self.global_pool(global_repr)
#         return global_repr

# # drop = 0.1 
# # drop = 0.1 + attention_weights √


# # better! 0.85~
# class OptimizedMHA4(nn.Module):
#     def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
#         super(OptimizedMHA4, self).__init__()
#         self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True,dropout=dropout_rate)
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.scale = embed_dim ** 0.5
#         self.pos_encoding = PositionalEncoding(embed_dim)
#         self.fc = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.Dropout(dropout_rate)  # 添加Dropout
#         )
        
#         self.dynamic_pooling = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.Tanh(),
#             nn.Dropout(dropout_rate), 
#             nn.Linear(embed_dim, 1),
#             nn.Softmax(dim=1)
#         )
        
#         self.global_pool = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim // 2),  # 降低中间维度
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),  # 添加Dropout
#             nn.Linear(embed_dim // 2, embed_dim)
#         )

#     def forward(self, x, key_padding_mask):
#         x = F.dropout(x, p=0.1, training=self.training)
#         x = x * self.scale
#         x = self.pos_encoding(x)
#         attn_output, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
#         attn_output = F.dropout(attn_output, p=0.1, training=self.training) 
#         x = self.norm1(x + attn_output)
#         fc_output = self.fc(x)
#         x = self.norm2(x + fc_output)
#         attn_weights = self.dynamic_pooling(x)
#         global_repr = torch.sum(attn_weights * x, dim=1)
#         global_repr = self.global_pool(global_repr)
#         return global_repr
    
# # 1 layers：Precision: 0.8670, Recall: 0.9264, F1 Score: 0.8957    0.45s
# # 2 layers：Precision: 0.8821, Recall: 0.9351, F1 Score: 0.9079   0.55s
# # 3 layers：Precision: 0.8885, Recall: 0.9318, F1 Score: 0.59s
# # 4 layers: Precision: 0.8949, Recall: 0.9319, F1 Score: 0.9131 # 00:59

# class OptimizedMHA5(nn.Module):
#     def __init__(self, embed_dim, num_heads, dropout_rate=0.1,num_layers=2):
#         super(OptimizedMHA5, self).__init__()
#         # self.linear = nn.Linear(embed_dim, embed_dim)
#         self.mha_layers = nn.ModuleList([
#             nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout_rate)
#             for _ in range(num_layers)
#         ])
#         self.norm_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])

#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.scale = embed_dim ** 0.5
#         self.pos_encoding = PositionalEncoding(embed_dim)
#         self.fc = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.Dropout(dropout_rate)  # 添加Dropout
#         )
        
#         self.dynamic_pooling = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.Tanh(),
#             nn.Dropout(dropout_rate), 
#             nn.Linear(embed_dim, 1),
#             nn.Softmax(dim=1)
#         )
        
#         self.global_pool = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim // 2),  # 降低中间维度
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),  # 添加Dropout
#             nn.Linear(embed_dim // 2, embed_dim)
#         )

#     def forward(self, x, key_padding_mask):
#         x = F.dropout(x, p=0.1, training=self.training)
#         x = x * self.scale
#         x = self.pos_encoding(x)
#         for layer, norm in zip(self.mha_layers, self.norm_layers):
#             attn_output, _ = layer(x, x, x, key_padding_mask=key_padding_mask)
#             attn_output = F.dropout(attn_output, p=0.1, training=self.training) 
#             x = norm(x + attn_output)  
#         fc_output = self.fc(x)
#         x = self.norm2(x + fc_output)
#         attn_weights = self.dynamic_pooling(x)
#         global_repr = torch.sum(attn_weights * x, dim=1)
#         global_repr = self.global_pool(global_repr)
#         return global_repr


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding2(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        # Create positional encoding matrix
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # Add positional encoding to input
        # Adjust for different input shapes
        if x.dim() == 3:  # [batch, seq_len, dim]
            return x + self.pe[:x.size(1), :]
        elif x.dim() == 4:  # [batch, n, seq_len, dim]
            return x + self.pe[:x.size(2), :].unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

class BatchDialogueSimilarityModel(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1, num_layers=2, temperature=0.07, no_match_threshold=0.3, similarity_threshold=0.5):
        super(BatchDialogueSimilarityModel, self).__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature  # Temperature parameter for InfoNCE loss
        self.no_match_threshold = no_match_threshold  # Threshold for determining "no match"
        self.similarity_threshold = similarity_threshold  # Threshold for minimal acceptable similarity
        
        # Multi-head attention layers
        self.mha_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout_rate)
            for _ in range(num_layers)
        ])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        
        self.norm_final = nn.LayerNorm(embed_dim)
        self.scale = embed_dim ** 0.5
        self.pos_encoding = PositionalEncoding2(embed_dim)
        
        # Feedforward networks
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Dynamic attention pooling
        self.dynamic_pooling = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate), 
            nn.Linear(embed_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Global representation enhancement
        self.global_pool = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # Cross-attention for comparing query with candidates
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True, 
            dropout=dropout_rate
        )
        
        # 精简版关系感知无匹配分类器
        # 只使用关键统计特征和较少的层数
        self.no_match_classifier = nn.Sequential(
            nn.Linear(5, 16),  # 只使用5个统计特征
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def _reshape_for_batch_processing(self, x, masks=None):
        """
        Reshape input tensors for batch processing
        [b, n, seq_len, dim] -> [b*n, seq_len, dim]
        """
        b, n, seq_len, dim = x.shape
        x_reshaped = x.view(b * n, seq_len, dim)
        
        if masks is not None:
            masks_reshaped = masks.view(b * n, seq_len)
            return x_reshaped, masks_reshaped
        
        return x_reshaped, None

    def encode_dialogue_batch(self, x, mask=None):
        """
        Encode a batch of dialogue sequences into contextualized representations
        
        Args:
            x: Tensor of shape [b*n, seq_len, dim]
            mask: Tensor of shape [b*n, seq_len] or None
            
        Returns:
            global_repr: Tensor of shape [b*n, dim]
            contextualized: Tensor of shape [b*n, seq_len, dim]
        """
        x = F.dropout(x, p=0.1, training=self.training)
        x = x * self.scale
        x = self.pos_encoding(x)
        
        # Apply multi-head attention layers
        for layer, norm in zip(self.mha_layers, self.norm_layers):
            attn_output, _ = layer(x, x, x, key_padding_mask=mask)
            attn_output = F.dropout(attn_output, p=0.1, training=self.training) 
            x = norm(x + attn_output)  
        
        # Apply feedforward network
        fc_output = self.fc(x)
        x = self.norm_final(x + fc_output)
        
        # Dynamic pooling to get weights for each token
        attn_weights = self.dynamic_pooling(x)
        
        # Weighted sum to get global representation
        global_repr = torch.sum(attn_weights * x, dim=1)
        global_repr = self.global_pool(global_repr)
        
        return global_repr, x
        
    def forward(self, dialogues, masks=None):
        """
        Process a batch of dialogues and compute similarities
        
        Args:
            dialogues: Tensor of shape [b, n, seq_len, dim]
                where n >= 2, with the first item being the query
                and the rest being the candidates
            masks: Tensor of shape [b, n, seq_len] or None
                
        Returns:
            similarities: Tensor of shape [b, n]
            - For each query, the first position (index 0) contains the "no match" score
            - The remaining positions contain similarity scores for each candidate
        """
        b, n, seq_len, dim = dialogues.shape
        assert n >= 2, "Need at least one query and one candidate per batch"
        
        # Reshape for batch processing
        dialogues_reshaped, masks_reshaped = self._reshape_for_batch_processing(dialogues, masks)
        
        # Encode all dialogues at once
        global_reprs, contextualized = self.encode_dialogue_batch(dialogues_reshaped, masks_reshaped)
        
        # Reshape back to [b, n, dim] for global representations
        global_reprs = global_reprs.view(b, n, dim)
        
        # Split query and candidates
        query_repr = global_reprs[:, 0]  # [b, dim]
        candidate_reprs = global_reprs[:, 1:]  # [b, n-1, dim]
        
        # Calculate similarities
        query_norm = F.normalize(query_repr, p=2, dim=1).unsqueeze(1)  # [b, 1, dim]
        candidate_norm = F.normalize(candidate_reprs, p=2, dim=2)  # [b, n-1, dim]
        
        # Compute similarity matrix: [b, 1, dim] × [b, dim, n-1] -> [b, 1, n-1]
        cosine_similarities = torch.bmm(query_norm, candidate_norm.transpose(1, 2)).squeeze(1)  # [b, n-1]
        
        # 计算关系特征（只使用核心统计特征）
        max_sim, _ = torch.max(cosine_similarities, dim=1)  # [b]
        mean_sim = torch.mean(cosine_similarities, dim=1)  # [b]
        min_sim, _ = torch.min(cosine_similarities, dim=1)  # [b]
        sim_std = torch.std(cosine_similarities, dim=1)  # [b]
        
        # 计算相似度差异（最高和次高相似度之间的差距）
        sorted_sims, _ = torch.sort(cosine_similarities, dim=1, descending=True)
        if sorted_sims.size(1) > 1:
            sim_gap = sorted_sims[:, 0] - sorted_sims[:, 1]  # [b]
        else:
            sim_gap = torch.zeros_like(max_sim)  # 如果只有一个候选项
        
        # 构建紧凑的特征向量（不包含查询表示）
        relation_features = torch.stack([
            max_sim,       # 最高相似度
            mean_sim,      # 平均相似度
            min_sim,       # 最低相似度
            sim_std,       # 相似度标准差
            sim_gap        # 最高和次高相似度之间的差距
        ], dim=1)  # [b, 5]
        
        # 预测无匹配概率
        no_match_scores = self.no_match_classifier(relation_features).squeeze(-1)  # [b]
        
        # 创建包含无匹配得分的相似度向量
        similarities = torch.cat([no_match_scores.unsqueeze(1), cosine_similarities], dim=1)  # [b, n]
        
        return similarities
    
    def compute_loss(self, similarities, labels):
        """
        Compute loss considering both match and no-match cases
        
        Args:
            similarities: Tensor of shape [b, n]
                First position (index 0) contains the "no match" score
                Remaining positions contain candidate similarity scores
            labels: Tensor of shape [b]
                Contains indices of positive samples or -1 if no match
                
        Returns:
            loss: Scalar loss value
        """
        batch_size = similarities.size(0)
        n_candidates = similarities.size(1) - 1  # Subtract 1 for "no match" score
        
        # Create target distribution:
        # 1. For no-match queries (label == -1), position 0 should be 1
        # 2. For match queries, position label+1 should be 1
        
        # Initialize target tensor with zeros
        target = torch.zeros_like(similarities)
        
        # Set target values
        for i in range(batch_size):
            if labels[i] == -1:
                # No match case
                target[i, 0] = 1.0
            else:
                # Match case (add 1 to label to account for "no match" position)
                target[i, labels[i] + 1] = 1.0
        
        # Apply temperature scaling
        logits = similarities / self.temperature
        
        # Compute cross-entropy loss
        log_probs = F.log_softmax(logits, dim=1)
        loss = -torch.sum(target * log_probs) / batch_size
        
        return loss
    
    def compute_similarity(self, query_dialogues, candidate_dialogues, query_masks=None, candidate_masks=None):
        """
        Compute similarity between query and candidate dialogues (for inference)
        
        Args:
            query_dialogues: Tensor of shape [b, 1, seq_len, dim]
            candidate_dialogues: Tensor of shape [b, m, seq_len, dim]
            query_masks: Tensor of shape [b, 1, seq_len] or None
            candidate_masks: Tensor of shape [b, m, seq_len] or None
            
        Returns:
            similarities: Tensor of shape [b, m+1]
                - First position contains "no match" score
                - Remaining positions contain similarity scores with candidates
            match_decision: Boolean tensor of shape [b]
                - True if the model predicts a match exists
                - False if the model predicts no match
        """
        b, _, seq_len, dim = query_dialogues.shape
        _, m, _, _ = candidate_dialogues.shape
        
        # Combine query and candidates
        combined_dialogues = torch.cat([query_dialogues, candidate_dialogues], dim=1)  # [b, m+1, seq_len, dim]
        
        # Combine masks if they exist
        combined_masks = None
        if query_masks is not None and candidate_masks is not None:
            combined_masks = torch.cat([query_masks, candidate_masks], dim=1)  # [b, m+1, seq_len]
        
        # Get representations
        combined_dialogues_reshaped, combined_masks_reshaped = self._reshape_for_batch_processing(
            combined_dialogues, combined_masks)
        
        global_reprs, _ = self.encode_dialogue_batch(combined_dialogues_reshaped, combined_masks_reshaped)
        global_reprs = global_reprs.view(b, m+1, dim)
        
        # Split query and candidates
        query_repr = global_reprs[:, 0]  # [b, dim]
        candidate_reprs = global_reprs[:, 1:]  # [b, m, dim]
        
        # Calculate similarities
        query_norm = F.normalize(query_repr, p=2, dim=1).unsqueeze(1)  # [b, 1, dim]
        candidate_norm = F.normalize(candidate_reprs, p=2, dim=2)  # [b, m, dim]
        
        cosine_similarities = torch.bmm(query_norm, candidate_norm.transpose(1, 2)).squeeze(1)  # [b, m]
        
        # 计算关系特征
        max_sim, _ = torch.max(cosine_similarities, dim=1)  # [b]
        mean_sim = torch.mean(cosine_similarities, dim=1)  # [b]
        min_sim, _ = torch.min(cosine_similarities, dim=1)  # [b]
        sim_std = torch.std(cosine_similarities, dim=1)  # [b]
        
        # 计算相似度差异
        sorted_sims, _ = torch.sort(cosine_similarities, dim=1, descending=True)
        if sorted_sims.size(1) > 1:
            sim_gap = sorted_sims[:, 0] - sorted_sims[:, 1]  # [b]
        else:
            sim_gap = torch.zeros_like(max_sim)  # 如果只有一个候选项
        
        # 构建紧凑的特征向量
        relation_features = torch.stack([
            max_sim,       # 最高相似度
            mean_sim,      # 平均相似度
            min_sim,       # 最低相似度
            sim_std,       # 相似度标准差
            sim_gap        # 最高和次高相似度之间的差距
        ], dim=1)  # [b, 5]
        
        # 预测无匹配概率
        no_match_scores = self.no_match_classifier(relation_features).squeeze(-1)  # [b]
        
        # 组合无匹配得分和候选项相似度
        similarities = torch.cat([no_match_scores.unsqueeze(1), cosine_similarities], dim=1)  # [b, m+1]
        
        # 使用多重标准确定是否有匹配
        # Get indices of highest similarity
        max_sim_idx = torch.argmax(similarities, dim=1)  # [b]
        
        # 判断是否存在匹配
        match_decision = torch.zeros(b, dtype=torch.bool, device=similarities.device)
        
        for i in range(b):
            # 无匹配条件:
            # 1. 无匹配得分最高 或
            # 2. 最高相似度低于阈值 或
            # 3. 最高相似度与次高相似度差距太小且相似度不够高
            if (max_sim_idx[i] == 0 or 
                max_sim[i] < self.similarity_threshold or
                (sim_gap[i] < 0.1 and max_sim[i] < 0.7)):
                match_decision[i] = False
            else:
                match_decision[i] = True
        
        return similarities, match_decision


# Example usage
def train_example():
    # Setup parameters
    embed_dim = 768
    num_heads = 8
    batch_size = 4
    n_samples = 5  # 1 query + 4 candidates per batch
    dialogue_len = 10
    
    # Create model
    model = BatchDialogueSimilarityModel(embed_dim=embed_dim, num_heads=num_heads)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop example
    for epoch in range(3):
        # Create dummy input data: [b, n, seq_len, dim]
        dialogues = torch.randn(batch_size, n_samples, dialogue_len, embed_dim)
        
        # Create dummy masks (True = padding tokens)
        masks = torch.zeros(batch_size, n_samples, dialogue_len, dtype=torch.bool)
        for i in range(batch_size):
            for j in range(n_samples):
                # Variable length padding
                padding_start = 7 + (i + j) % 3
                if padding_start < dialogue_len:
                    masks[i, j, padding_start:] = True
        
        # Create dummy labels (some with matches, some without)
        # Label -1 means no match, other values are indices of positive samples
        labels = torch.tensor([1, -1, 2, -1], dtype=torch.long)
        
        # Forward pass
        similarities = model(dialogues, masks)
        
        # Compute loss
        loss = model.compute_loss(similarities, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        
        # Print similarity matrix for the first batch
        if epoch == 2:
            print(f"Similarities (batch 0): {similarities[0]}")
    
    return model

# Inference example
def inference_example(model):
    # Setup parameters
    embed_dim = 768
    batch_size = 2
    dialogue_len = 10
    n_candidates = 3
    
    # Create dummy query data: [b, 1, seq_len, dim]
    query_dialogues = torch.randn(batch_size, 1, dialogue_len, embed_dim)
    
    # Create dummy candidate data: [b, m, seq_len, dim]
    candidate_dialogues = torch.randn(batch_size, n_candidates, dialogue_len, embed_dim)
    
    # Create masks
    query_masks = torch.zeros(batch_size, 1, dialogue_len, dtype=torch.bool)
    query_masks[:, :, 8:] = True  # Last 2 tokens are padding
    
    candidate_masks = torch.zeros(batch_size, n_candidates, dialogue_len, dtype=torch.bool)
    for i in range(batch_size):
        for j in range(n_candidates):
            candidate_masks[i, j, 7+j:] = True  # Variable padding
    
    # Compute similarities
    with torch.no_grad():
        similarities, match_decision = model.compute_similarity(
            query_dialogues, candidate_dialogues, query_masks, candidate_masks
        )
    
    print(f"Inference similarities: {similarities}")
    print(f"Match decision: {match_decision}")
    
    # Find most similar candidates for each query (if match exists)
    most_similar_idx = torch.zeros(batch_size, dtype=torch.long)
    for i in range(batch_size):
        if match_decision[i]:
            # Skip the no-match score at index 0
            most_similar_idx[i] = torch.argmax(similarities[i, 1:]) + 1
        else:
            most_similar_idx[i] = 0  # 0 indicates no match
    
    print(f"Most similar candidate indices (0 means no match): {most_similar_idx}")
    
    return similarities, match_decision

if __name__ == "__main__":
    model = train_example()
    inference_example(model)