import torch
import torch.nn as nn

# 定义模型类
class MHAModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, dropout=0.1):
        super(MHAModel, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.residual = True
        # 添加layer norm
        self.norm = nn.LayerNorm(embed_dim)
        # 可学习的混合参数
        self.alpha = nn.Parameter(torch.FloatTensor([0.5]))

    def forward(self, x, key_padding_mask):
        # Multi-Head Attention with padding mask
        mask = ~key_padding_mask.unsqueeze(-1)
        attn_output, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)  # Self-attention
        if self.residual:
            attn_output = attn_output + x
        attn_output = self.norm(attn_output)
        print("attn_output shape after norm:", attn_output.shape)
        last_indices = (~key_padding_mask).sum(dim=1) - 1
        attn_output = attn_output[torch.arange(attn_output.size(0)), last_indices].unsqueeze(1)
        print("attn_output shape after indexing:", attn_output.shape)
        # 在序列维度上取平均得到全局表示
        mean_pooled = torch.mean(x, dim=1, keepdim=True)  # [batch_size, 1, embed_dim]
        print("mean_pooled shape:", mean_pooled.shape)
        output = self.alpha * attn_output + (1-self.alpha) * mean_pooled  # [batch_size, 1, embed_dim]
        print("output shape after weighted sum:", output.shape)
        return output.squeeze(1)

# 测试代码
def test_model():
    batch_size = 4
    seq_len = 10
    input_dim = 64
    embed_dim = 64
    num_heads = 4

    # 初始化模型
    model = MHAModel(input_dim=input_dim, embed_dim=embed_dim, num_heads=num_heads)

    # 随机生成输入数据
    x = torch.randn(batch_size, seq_len, embed_dim)  # [batch_size, seq_len, embed_dim]
    key_padding_mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool)  # [batch_size, seq_len]

    print("Input shape:", x.shape)
    print("Key padding mask shape:", key_padding_mask.shape)

    # 前向传播
    output = model(x, key_padding_mask)

    print("Final output shape:", output.shape)

# 运行测试
test_model()
