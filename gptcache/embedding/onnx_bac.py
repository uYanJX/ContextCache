import numpy as np
import torch

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

class Onnx(BaseEmbedding):
    """Generate text embedding for given text using ONNX Model.

    Example:
        .. code-block:: python

            from gptcache.embedding import Onnx

            test_sentence = 'Hello, world.'
            encoder = Onnx(model='GPTCache/paraphrase-albert-onnx')
            embed = encoder.to_embeddings(test_sentence)
    """

    def __init__(self, model="yilunzhang/all-mpnet-base-v2-onnx"):
        tokenizer_name = "yilunzhang/all-mpnet-base-v2-onnx"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = model
        onnx_model_path = hf_hub_download(repo_id=model, filename="model.onnx")
        self.ort_session = onnxruntime.InferenceSession(onnx_model_path)
        config = AutoConfig.from_pretrained(
            "yilunzhang/all-mpnet-base-v2-onnx"
        )
        
        self.__dimension = config.hidden_size
        
        # 获取模型的输入名称以确定需要哪些参数
        self.input_names = [input.name for input in self.ort_session.get_inputs()]

    def to_embeddings(self, data, **_):
        """Generate embedding given text input.

        :param data: text in string.
        :type data: str

        :return: a text embedding in shape of (dim,).
        """
        encoded_text = self.tokenizer(
            data, 
            padding=True, 
            truncation=True, 
            return_tensors="np",
            max_length=512
        )

        ort_inputs = {}
        if "input_ids" in self.input_names:
            ort_inputs["input_ids"] = encoded_text["input_ids"].astype("int64")
        if "attention_mask" in self.input_names:
            ort_inputs["attention_mask"] = encoded_text["attention_mask"].astype("int64")
        if "token_type_ids" in self.input_names and "token_type_ids" in encoded_text:
            ort_inputs["token_type_ids"] = encoded_text["token_type_ids"].astype("int64")

        ort_outputs = self.ort_session.run(None, ort_inputs)
        ort_feat = ort_outputs[0]
        
        # 使用mean pooling处理输出
        emb = self.mean_pooling(ort_feat, ort_inputs["attention_mask"])
        return emb.flatten()

    def mean_pooling(self, model_output, attention_mask):
        """Mean Pooling - Take attention mask into account for correct averaging"""
        token_embeddings = model_output  # 第一个元素包含所有token embeddings
        input_mask_expanded = (
            np.expand_dims(attention_mask, -1)
            .repeat(token_embeddings.shape[-1], -1)
            .astype(float)
        )
        sentence_embs = np.sum(token_embeddings * input_mask_expanded, 1) / np.maximum(
            input_mask_expanded.sum(1), 1e-9
        )
        return sentence_embs

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
        return self.mean_pooling(token_embeddings, attention_mask)

    @property
    def dimension(self):
        """Embedding dimension.

        :return: embedding dimension
        """
        return self.__dimension