import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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

     
class OptimizedMHA(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super(OptimizedMHA, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.scale = embed_dim ** 0.5
        self.norm2 = nn.LayerNorm(embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.global_pool = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

    def forward(self, x, key_padding_mask):
        # Multi-Head Attention with padding mask
        x = x * self.scale
        x = self.pos_encoding(x)
        attn_output, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        # Residual connection and normalization
        x = self.norm1(x + attn_output)
        # Fully connected transformation
        fc_output = self.fc(x)
        # Residual connection and normalization
        x = self.norm2(x + fc_output)
        # Global aggregation with masking (ensuring padded positions are excluded)
        mask = ~key_padding_mask.unsqueeze(-1)
        masked_output = x * mask
        # Weighted global representation
        global_repr = (masked_output.sum(dim=1) / mask.sum(dim=1))
        global_repr = self.global_pool(global_repr)
        return global_repr 

# best 0.83~ 
class OptimizedMHA2(nn.Module):
    def __init__(self,  embed_dim, num_heads, dropout_rate=0.1):
        super(OptimizedMHA2, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True,dropout=0.1)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.scale = embed_dim ** 0.5
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout_rate)  # 添加Dropout
        )
        self.global_pool = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),  # 降低中间维度
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 添加Dropout
            nn.Linear(embed_dim // 2, embed_dim)
        )
        self.attention_weights = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x, key_padding_mask):
        x = F.dropout(x, p=0.1, training=self.training)
        x = x * self.scale
        x = self.pos_encoding(x)
        attn_output, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        attn_output = F.dropout(attn_output, p=0.1, training=self.training) 
        x = self.norm1(x + attn_output)
        fc_output = self.fc(x)
        x = self.norm2(x + fc_output)
        attn_weights = self.attention_weights(x)
        global_repr = torch.sum(attn_weights * x, dim=1)
        global_repr = self.global_pool(global_repr)
        return global_repr

# drop = 0.1 
# drop = 0.1 + attention_weights √


# better! 0.85~
class OptimizedMHA4(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super(OptimizedMHA4, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True,dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
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
            nn.Linear(embed_dim, embed_dim // 2),  # 降低中间维度
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 添加Dropout
            nn.Linear(embed_dim // 2, embed_dim)
        )

    def forward(self, x, key_padding_mask):
        x = F.dropout(x, p=0.1, training=self.training)
        x = x * self.scale
        x = self.pos_encoding(x)
        attn_output, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        attn_output = F.dropout(attn_output, p=0.1, training=self.training) 
        x = self.norm1(x + attn_output)
        fc_output = self.fc(x)
        x = self.norm2(x + fc_output)
        attn_weights = self.dynamic_pooling(x)
        global_repr = torch.sum(attn_weights * x, dim=1)
        global_repr = self.global_pool(global_repr)
        return global_repr
    
# 1 layers：Precision: 0.8670, Recall: 0.9264, F1 Score: 0.8957    0.45s
# 2 layers：Precision: 0.8821, Recall: 0.9351, F1 Score: 0.9079   0.55s
# 3 layers：Precision: 0.8885, Recall: 0.9318, F1 Score: 0.59s
# 4 layers: Precision: 0.8949, Recall: 0.9319, F1 Score: 0.9131 # 00:59

class OptimizedMHA5(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1,num_layers=2):
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
            nn.Linear(embed_dim, embed_dim // 2),  # 降低中间维度
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 添加Dropout
            nn.Linear(embed_dim // 2, embed_dim)
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
    
################################################### 
# new idea
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
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1, num_layers=2, temperature=0.07):
        super(BatchDialogueSimilarityModel, self).__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature  # Temperature parameter for InfoNCE loss
        
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
        Process a batch of dialogues and compute similarities using InfoNCE loss framework
        
        Args:
            dialogues: Tensor of shape [b, n, seq_len, dim]
                where n >= 2, with the first item being the query
                and the rest being the candidates
            masks: Tensor of shape [b, n, seq_len] or None
                
        Returns:
            similarities: Tensor of shape [b, n-1]
            loss: InfoNCE loss
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
        similarities = torch.bmm(query_norm, candidate_norm.transpose(1, 2)).squeeze(1)  # [b, n-1]
        
        # Calculate InfoNCE loss
        # Assumption: The first candidate (index 0) is the positive sample
        labels = torch.zeros(b, dtype=torch.long, device=dialogues.device)
        
        # Apply temperature scaling
        logits = similarities / self.temperature
        loss = F.cross_entropy(logits, labels)
        
        return similarities, loss
    
    def compute_similarity(self, query_dialogues, candidate_dialogues, query_masks=None, candidate_masks=None):
        """
        Compute similarity between query and candidate dialogues (for inference)
        
        Args:
            query_dialogues: Tensor of shape [b, 1, seq_len, dim]
            candidate_dialogues: Tensor of shape [b, m, seq_len, dim]
            query_masks: Tensor of shape [b, 1, seq_len] or None
            candidate_masks: Tensor of shape [b, m, seq_len] or None
            
        Returns:
            similarities: Tensor of shape [b, m]
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
        
        similarities = torch.bmm(query_norm, candidate_norm.transpose(1, 2)).squeeze(1)  # [b, m]
        
        return similarities

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
        
        # Forward pass
        similarities, loss = model(dialogues, masks)
        
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
        similarities = model.compute_similarity(
            query_dialogues, candidate_dialogues, query_masks, candidate_masks
        )
    
    print(f"Inference similarities: {similarities}")
    
    # Find most similar candidates for each query
    most_similar_idx = torch.argmax(similarities, dim=1)
    print(f"Most similar candidate indices: {most_similar_idx}")
    
    return similarities, most_similar_idx

if __name__ == "__main__":
    model = train_example()
    inference_example(model)