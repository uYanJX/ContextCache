import json
import random
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from torch.utils.data import Dataset
import re
from sklearn.metrics.pairwise import cosine_similarity
import os
from tqdm import tqdm
import ipdb
import torch.multiprocessing as mp
from functools import partial

# export HF_ENDPOINT=https://hf-mirror.com

# 定义清理编号的函数
def clean_numbering(origin):
    sentences = re.split(r'^\d+\.\s|(?:\n)\d+\.\s', origin, flags=re.MULTILINE)
    return [re.sub(r'^\d+\.\s*', '', sentence) for sentence in sentences if sentence.strip()]

class QueryDataset(Dataset):
    def __init__(self, 
                 pos_file_path, 
                 neg_file_path,
                 max_seq_length=10, 
                 sentence_model_name="all-mpnet-base-v2", # "paraphrase-albert-small-v2", # "all-mpnet-base-v2", 
                 batch_size=512,
                 log=None,
                 device=None
                 ):
        self.logger = log
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_seq_length = max_seq_length
        self.data_pairs = []
        # 尝试加载缓存
        cache_path = f"/data/home/Jianxin/MyProject/ContextCache/cache/test/albert_{os.path.basename(pos_file_path).split('.')[0]}.pkl"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        if os.path.exists(cache_path):
            self.data_pairs = torch.load(cache_path)
            self.logger.info(f"Loaded data from cache: {cache_path}")
            data_len = len(self.data_pairs)
            pos_count = sum(label for _, _, label in self.data_pairs)
            self.logger.info(f"Loaded {data_len} data pairs, {pos_count} positive pairs, {data_len-pos_count} negative pairs")
            return

        self.sentence_model = SentenceTransformer(sentence_model_name).to(self.device)
    
        # 加载数据并预处理
        pos_data = [json.loads(line) for line in open(pos_file_path)]
        neg_data = [json.loads(line) for line in open(neg_file_path)]
        
        # 收集所有句子并编码
        all_sentences = set()
        for item in pos_data:
            for key in ["original", "variations", "simplified", "expanded"]:
                all_sentences.update(clean_numbering(item[key]))
        for item in neg_data:
            all_sentences.update(clean_numbering(item["neg"]))
            
        # 批量编码
        embeddings = self.parallel_encode(all_sentences, batch_size)
        
        # 生成正样本
        for item in pos_data:
            orig = np.array([embeddings[s] for s in clean_numbering(item["original"])])
            # if not orig:
            #     continue
                
            for key in ["variations", "simplified", "expanded"]:
                sentences = clean_numbering(item[key])
                self.data_pairs.append((orig, np.array([embeddings[s] for s in sentences]), 1))
                    
        # 生成负样本
        for i, item in enumerate(neg_data):
            orig = clean_numbering(item["original"])
            neg = clean_numbering(item["neg"])
                
            orig_enc = np.array([embeddings[s] for s in orig])
            neg_enc = np.array([embeddings[s] for s in neg])
            
            # 基本负样本
            self.data_pairs.append((orig_enc, neg_enc, 0))
            
            # 打乱原样本生成负样本
            if len(orig) > 1:
                shuffled_orig = orig_enc.copy()
                idx = random.randint(0, len(orig_enc)-2)
                shuffled_orig[idx], shuffled_orig[-1] = shuffled_orig[-1], shuffled_orig[idx]
                self.data_pairs.append((orig_enc, shuffled_orig, 0))
            
            # 随机选择其他样本作为负样本
            other_idx = random.choice([j for j in range(len(neg_data)) if j != i])
            other_sentences = clean_numbering(neg_data[other_idx]["original"])
            self.data_pairs.append((orig_enc, np.array([embeddings[s] for s in other_sentences]), 0))
        
        data_len = len(self.data_pairs)
        pos_count = sum(label for _, _, label in self.data_pairs)
        self.logger.info(f"Loaded {data_len} data pairs, {pos_count} positive pairs, {data_len-pos_count} negative pairs")
        
        # 保存缓存
        torch.save(self.data_pairs, cache_path)

    def __getitem__(self, idx):
        s1, s2, label = self.data_pairs[idx]
        return (torch.tensor(self.pad_sequence(s1), dtype=torch.float32, device=self.device),
                torch.tensor(self.pad_sequence(s2), dtype=torch.float32, device=self.device),
                torch.tensor(label, device=self.device))
    
    def pad_sequence(self, sequence):
        if len(sequence) > self.max_seq_length:
            return sequence[:self.max_seq_length]
        padding = np.zeros((self.max_seq_length - len(sequence), sequence.shape[1]))
        return np.vstack([sequence, padding])
        
    def __len__(self):
        return len(self.data_pairs)

    def encode_batch(self, batch_and_gpu):
        """
        对单个批次进行编码，并在指定的 GPU 上运行。
        """
        sentences, gpu_id = batch_and_gpu
        device = torch.device(gpu_id)  # 分配到指定 GPU
        return self.sentence_model.encode(
            sentences,
            convert_to_numpy=True,
            device=device,
            show_progress_bar=False
        )

    def parallel_encode(self, all_sentences, batch_size, num_workers=6, gpus=None):
        """
        使用多进程并行对句子进行编码，每个进程分配到一个特定的 GPU。

        Args:
            all_sentences (list): 需要编码的所有句子。
            batch_size (int): 每个批次的大小。
            num_workers (int): 使用的进程数量。
            gpus (list): 可用的 GPU 列表，如 ["cuda:1", "cuda:2", ..., "cuda:7"]。

        Returns:
            dict: 一个字典，句子作为键，嵌入向量作为值。
        """
        embeddings = {}
        sentences_list = list(all_sentences)
        total_batches = (len(sentences_list) + batch_size - 1) // batch_size

        # 将数据分成多个批次
        batches = [
            sentences_list[i:i + batch_size]
            for i in range(0, len(sentences_list), batch_size)
        ]

        # 确保有足够的 GPU 分配
        if gpus is None:
            gpus = [f"cuda:{i+2}" for i in range(num_workers)]
        assert len(gpus) >= num_workers, "Number of GPUs must be at least equal to num_workers"

        # 设置多进程上下文
        ctx = mp.get_context('spawn')

        with ctx.Pool(num_workers) as pool:
            # 为每个批次分配 GPU（轮询分配 GPU）
            gpu_assignments = [gpus[i % len(gpus)] for i in range(len(batches))]
            tasks = zip(batches, gpu_assignments)

            # 并行处理批次
            results = list(tqdm(
                pool.imap(partial(self.encode_batch), tasks),
                total=total_batches,
                desc="parallel encoding sentences"
            ))

        # 合并结果
        for batch, encodings in zip(batches, results):
            embeddings.update(dict(zip(batch, encodings)))

        return embeddings

if __name__ == "__main__":
    # 设置日志
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 创建测试数据
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    # 创建测试用的正样本数据
    pos_data = [
        {
            "original": "1. 这是第一句话\n2. 这是第二句话1",
            "variations": "1. 这是第一个变体\n2. 这是第二个变体1",
            "simplified": "1. 简化的第一句\n2. 简化的第二句1",
            "expanded": "1. 扩展的第一句\n2. 扩展的第二句1"
        },
        {
            "original": "1. 这是第一句话\n2. 这是第二句话2",
            "variations": "1. 这是第一个变体\n2. 这是第二个变体2",
            "simplified": "1. 简化的第一句\n2. 简化的第二句2",
            "expanded": "1. 扩展的第一句\n2. 扩展的第二句2"
        },
        {
            "original": "1. 这是第一句话\n2. 这是第二句话3",
            "variations": "1. 这是第一个变体\n2. 这是第二个变体3",
            "simplified": "1. 简化的第一句\n2. 简化的第二句3",
            "expanded": "1. 扩展的第一句\n2. 扩展的第二句3"
        },
        {
            "original": "1. 这是第一句话\n2. 这是第二句话4",
            "variations": "1. 这是第一个变体\n2. 这是第二个变体4",
            "simplified": "1. 简化的第一句\n2. 简化的第二句4",
            "expanded": "1. 扩展的第一句\n2. 扩展的第二句4"
        },
        {
            "original": "1. 这是第一句话\n2. 这是第二句话5",
            "variations": "1. 这是第一个变体\n2. 这是第二个变体5",
            "simplified": "1. 简化的第一句\n2. 简化的第二句5",
            "expanded": "1. 扩展的第一句\n2. 扩展的第二句5"
        },
        {
            "original": "1. 这是第一句话\n2. 这是第二句话6",
            "variations": "1. 这是第一个变体\n2. 这是第二个变体6",
            "simplified": "1. 简化的第一句\n2. 简化的第二句6",
            "expanded": "1. 扩展的第一句\n2. 扩展的第二句6"
        },
        {
            "original": "1. 这是第一句话\n2. 这是第二句话7",
            "variations": "1. 这是第一个变体\n2. 这是第二个变体7",
            "simplified": "1. 简化的第一句\n2. 简化的第二句7",
            "expanded": "1. 扩展的第一句\n2. 扩展的第二句7"
        },
        {
            "original": "1. 这是第一句话\n2. 这是第二句话8",
            "variations": "1. 这是第一个变体\n2. 这是第二个变体8",
            "simplified": "1. 简化的第一句\n2. 简化的第二句8",
            "expanded": "1. 扩展的第一句\n2. 扩展的第二句8"
        },
        {
            "original": "1. 这是第一句话\n2. 这是第二句话9",
            "variations": "1. 这是第一个变体\n2. 这是第二个变体9",
            "simplified": "1. 简化的第一句\n2. 简化的第二句9",
            "expanded": "1. 扩展的第一句\n2. 扩展的第二句9"
        },
        {
            "original": "1. 这是第一句话\n2. 这是第二句话10",
            "variations": "1. 这是第一个变体\n2. 这是第二个变体10",
            "simplified": "1. 简化的第一句\n2. 简化的第二句10",
            "expanded": "1. 扩展的第一句\n2. 扩展的第二句10"
        }
    ]
    
    # 创建测试用的负样本数据
    neg_data = [
        {
            "original": "1. 原始的第一句\n2. 原始的第二句1",
            "neg": "1. 负面的第一句\n2. 负面的第二句1"
        },
        {
            "original": "1. 原始的第一句\n2. 原始的第二句2",
            "neg": "1. 负面的第一句\n2. 负面的第二句2"
        },
        {
            "original": "1. 原始的第一句\n2. 原始的第二句3",
            "neg": "1. 负面的第一句\n2. 负面的第二句3"
        },
        {
            "original": "1. 原始的第一句\n2. 原始的第二句4",
            "neg": "1. 负面的第一句\n2. 负面的第二句4"
        },
        {
            "original": "1. 原始的第一句\n2. 原始的第二句5",
            "neg": "1. 负面的第一句\n2. 负面的第二句5"
        }
    ]
    
    # 保存测试数据到临时文件
    pos_file = os.path.join(temp_dir, "test_pos.jsonl")
    neg_file = os.path.join(temp_dir, "test_neg.jsonl")
    
    with open(pos_file, 'w', encoding='utf-8') as f:
        for item in pos_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    with open(neg_file, 'w', encoding='utf-8') as f:
        for item in neg_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    try:
        # 测试数据集加载
        print("测试数据集加载...")
        dataset = QueryDataset(
            pos_file_path=pos_file,
            neg_file_path=neg_file,
            max_seq_length=5,
            batch_size=2,
            log=logger
        )
        print(f"数据集大小: {len(dataset)}")
        
        # 测试数据项获取
        print("\n测试数据项获取...")
        item = dataset[0]
        print(f"数据项类型: {type(item)}")
        print(f"输入1形状: {item[0].shape}")
        print(f"输入2形状: {item[1].shape}")
        print(f"标签值: {item[2]}")
        
        # 测试缓存加载
        print("\n测试缓存加载...")
        dataset2 = QueryDataset(
            pos_file_path=pos_file,
            neg_file_path=neg_file,
            log=logger
        )
        print(f"从缓存加载的数据集大小: {len(dataset2)}")
        
    finally:
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)
        print("\n清理完成：临时文件已删除")
