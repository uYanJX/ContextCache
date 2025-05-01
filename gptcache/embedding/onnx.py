import numpy as np

from gptcache.embedding.base import BaseEmbedding
from gptcache.utils import (
    import_onnxruntime,
    import_huggingface_hub,
    import_huggingface,
)

import_huggingface()
import_onnxruntime()
import_huggingface_hub()

from transformers import AutoTokenizer, AutoConfig  # pylint: disable=C0413
from huggingface_hub import hf_hub_download  # pylint: disable=C0413
import onnxruntime  # pylint: disable=C0413
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


class Onnx(BaseEmbedding):
    """Generate text embedding for given text using ONNX Model.

    Example:
        .. code-block:: python

            from gptcache.embedding import Onnx

            test_sentence = 'Hello, world.'
            encoder = Onnx(model='GPTCache/paraphrase-albert-onnx')
            embed = encoder.to_embeddings(test_sentence)
    """

    def __init__(self, model="GPTCache/paraphrase-albert-onnx"):
        tokenizer_name = "GPTCache/paraphrase-albert-small-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = model
        onnx_model_path = hf_hub_download(repo_id=model, filename="model.onnx")
        self.ort_session = onnxruntime.InferenceSession(onnx_model_path)
        config = AutoConfig.from_pretrained(
            "GPTCache/paraphrase-albert-small-v2"
        )
        
        # self.ContextModel= OptimizedMHA(embed_dim=768, num_heads=8)
        # self.ContextModel = self.ContextModel.to("cuda:4")
        # checkpoint = torch.load("/data/home/Jianxin/MyProject/ContextCache/results/exp_20250125_235628_4/best_model.pth")
        # self.ContextModel.load_state_dict(checkpoint)
        
        self.__dimension = config.hidden_size

    def to_embeddings(self, data, **_):
        """Generate embedding given text input.

        :param data: text in string.
        :type data: str

        :return: a text embedding in shape of (dim,).
        """
        encoded_text = self.tokenizer.encode_plus(data, padding="max_length")

        ort_inputs = {
            "input_ids": np.array(encoded_text["input_ids"]).astype("int64").reshape(1, -1),
            "attention_mask": np.array(encoded_text["attention_mask"]).astype("int64").reshape(1, -1),
            "token_type_ids": np.array(encoded_text["token_type_ids"]).astype("int64").reshape(1, -1),
        }

        ort_outputs = self.ort_session.run(None, ort_inputs)
        ort_feat = ort_outputs[0]
        emb = self.post_proc(ort_feat, ort_inputs["attention_mask"])
        return emb.flatten()

    def to_context_embeddings(self, datas, **_):
        """Generate context embedding given text input.

        :param datas: text in string.
        :type data: str

        :return: a text embedding in shape of (dim,).
        """
        embs = torch.stack([torch.from_numpy(self.to_embeddings(data)) for data in datas]).to("cuda:4").unsqueeze(0).to(torch.float32)
        # print(embs.shape)
        s_mask = embs.sum(dim=-1) == 0
        context_repr = self.ContextModel(embs, key_padding_mask=s_mask)
        return context_repr.squeeze(0).detach().cpu().numpy()
    
    def post_proc(self, token_embeddings, attention_mask):
        input_mask_expanded = (
            np.expand_dims(attention_mask, -1)
            .repeat(token_embeddings.shape[-1], -1)
            .astype(float)
        )
        sentence_embs = np.sum(token_embeddings * input_mask_expanded, 1) / np.maximum(
            input_mask_expanded.sum(1), 1e-9
        )
        return sentence_embs

    @property
    def dimension(self):
        """Embedding dimension.

        :return: embedding dimension
        """
        return self.__dimension
