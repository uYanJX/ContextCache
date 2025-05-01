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
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1,num_layers=2):
        super(OptimizedMHA, self).__init__()
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

ContextModel= OptimizedMHA(embed_dim=768, num_heads=8).to("cuda:4")
checkpoint = torch.load("/data/home/Jianxin/MyProject/ContextCache/results/exp_20250125_235628_4/best_model.pth")
ContextModel.load_state_dict(checkpoint)


# 创建一个与模型输入匹配的假输入 (dummy input)
dummy_input = torch.randn(1, 10, 768).to("cuda:4")  # 假设输入是一个批量大小为1，大小为768的向量
dummy_key_padding_mask = torch.ones(1, 10).to("cuda:4")

# 导出为 ONNX 格式
onnx_file_path = "context_model.onnx"
torch.onnx.export(
    ContextModel,         # 模型
    (dummy_input,dummy_key_padding_mask),               # 假输入
    onnx_file_path,            # 输出路径
    export_params=True,        # 导出模型的权重
    opset_version=13,          # 设置 ONNX 版本
    do_constant_folding=True,  # 是否进行常量折叠优化
    input_names=["input"],     # 输入名称
    output_names=["output"],   # 输出名称
    dynamic_axes={             # 可选，设置动态轴（如批量大小）
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print(f"Model has been converted to ONNX and saved as {onnx_file_path}")
