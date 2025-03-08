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
import logging
import torch.nn as nn
from embedding import Onnx

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
                 sentence_model_name="all-mpnet-base-v2", 
                 batch_size=512,
                 log=None,
                 is_train=True,
                 device=None
                 ):
        self.logger = log
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_seq_length = max_seq_length
        self.sentence_model = SentenceTransformer(sentence_model_name).to(self.device)
        # self.sentences_model = Onnx(model=sentence_model_name)
        
        self.data_pairs = []
        self.data_pairs_mean = []
        self.data_pairs_merge = []
        if is_train:
            self.logger.info(f"Loading data from train dataset")
        else:
            self.logger.info(f"Loading data from val dataset")
        
        # 尝试加载缓存
        cache_path = f"/data/home/Jianxin/MyProject/ContextCache/cache/test/mpnet_{os.path.basename(pos_file_path).split('.')[0]}.pkl"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        if os.path.exists(cache_path):
            self.data_pairs = torch.load(cache_path)
            self.logger.info(f"Loaded data from cache: {cache_path}")
            data_len = len(self.data_pairs)
            pos_count = sum(label for _, _, label in self.data_pairs)
            self.logger.info(f"Loaded {data_len} data pairs, {pos_count} positive pairs, {data_len-pos_count} negative pairs")
            return
            
        # 加载数据并预处理
        pos_data = [json.loads(line) for line in open(pos_file_path)]
        neg_data = [json.loads(line) for line in open(neg_file_path)]
        
        # 收集所有句子并编码
        all_sentences = set()
        for item in pos_data:
            for key in ["original", "variations", "simplified", "expanded"]:
                tmp = clean_numbering(item[key])
                all_sentences.update(tmp)
                merge_tmp = [" ".join(tmp)]
                all_sentences.update(merge_tmp)
                
        for item in neg_data:
            tmp1 = clean_numbering(item["original"])
            tmp2 = clean_numbering(item["neg"])
            all_sentences.update(tmp1)
            all_sentences.update(tmp2)
            merge_tmp1 = [" ".join(tmp1)]
            merge_tmp2 = [" ".join(tmp2)]
            all_sentences.update(merge_tmp1)
            all_sentences.update(merge_tmp2)
            
        # 批量编码
        embeddings = self.parallel_encode(all_sentences, batch_size)
        
        # 生成正样本
        for item in tqdm(pos_data):
            orig = np.array([embeddings[s] for s in clean_numbering(item["original"])])
            merge_orig = " ".join(clean_numbering(item["original"]))
            # if not orig:
            #     continue
            orig_mean = np.mean(orig, axis=0)
            orig_merge = embeddings[merge_orig]
            for key in ["variations", "simplified", "expanded"]:
                sentences = clean_numbering(item[key])
                merge_sentences = " ".join(sentences)
                st_encoded = np.array([embeddings[s] for s in sentences])
                
                self.data_pairs_merge.append((orig_merge, embeddings[merge_sentences], 1))
                self.data_pairs_mean.append((orig_mean, np.mean(st_encoded, axis=0), 1))
                self.data_pairs.append((orig, st_encoded , 1))
                    
        # 生成负样本
        for i, item in tqdm(enumerate(neg_data)):
            orig = clean_numbering(item["original"])
            neg = clean_numbering(item["neg"])
            # if not orig or not neg:
            #     continue
                
            orig_enc = np.array([embeddings[s] for s in orig])
            neg_enc = np.array([embeddings[s] for s in neg])
            
            
            # 基本负样本
            self.data_pairs.append((orig_enc, neg_enc, 0))
            self.data_pairs_mean.append((np.mean(orig_enc, axis=0), np.mean(neg_enc, axis=0), 0))
            self.data_pairs_merge.append((embeddings[" ".join(orig)], embeddings[" ".join(neg)], 0))
            
            # 打乱原样本生成负样本
            if len(orig) > 1:
                shuffled_orig = orig_enc.copy()
                idx = random.randint(0, len(orig_enc)-2)
                shuffled_orig[idx], shuffled_orig[-1] = shuffled_orig[-1], shuffled_orig[idx]
                self.data_pairs.append((orig_enc, shuffled_orig, 0))
                self.data_pairs_mean.append((np.mean(orig_enc, axis=0), np.mean(shuffled_orig, axis=0), 0))
                
                st_tmp = [s for s in orig]
                st_tmp[idx], st_tmp[-1] = st_tmp[-1], st_tmp[idx]
                new_orig = " ".join(st_tmp)
                new_orig_enc = self.sentence_model.encode(new_orig,show_progress_bar=False,convert_to_numpy=True)
                    
                self.data_pairs_merge.append((embeddings[" ".join(orig)], new_orig_enc, 0))
            
            # 随机选择其他样本作为负样本
            other_idx = random.choice([j for j in range(len(neg_data)) if j != i])
            other_sentences = clean_numbering(neg_data[other_idx]["original"])
            self.data_pairs.append((orig_enc, np.array([embeddings[s] for s in other_sentences]), 0))
            self.data_pairs_mean.append((np.mean(orig_enc, axis=0), np.mean([embeddings[s] for s in other_sentences], axis=0), 0))
            self.data_pairs_merge.append((embeddings[" ".join(orig)], embeddings[" ".join(other_sentences)], 0))
        
        
        data_len = len(self.data_pairs)
        pos_count = sum(label for _, _, label in self.data_pairs if label == 1)
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
        embeddings = self.sentence_model.encode(
            sentences,
            convert_to_numpy=True,
            device=device,
            show_progress_bar=False
        )
        
        # 清理 GPU 内存，防止泄漏
        torch.cuda.empty_cache()
        return embeddings

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

        # 清理 GPU 内存，防止泄漏
        torch.cuda.empty_cache()

        return embeddings

    
def evaluate_metrix(records):
    precision = records[0] / (records[0] + records[1]) if (records[0] + records[1]) > 0 else 0
    recall = records[0] / (records[0] + records[2]) if (records[0] + records[2]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

if __name__ == "__main__":
    log_file = "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    data = QueryDataset(
        "/data/home/Jianxin/MyProject/ContextCache/data/final/pos_dataset.jsonl",
        "/data/home/Jianxin/MyProject/ContextCache/data/final/neg_dataset.jsonl",
        log=logger,
        device="cuda:1"
        )

    # mean_tf = [0]*4
    # cos = nn.CosineSimilarity(dim=0)
    # for item in data.data_pairs_mean:
    #     s1,s2,label = item
    #     s1 = torch.from_numpy(s1)
    #     s2 = torch.from_numpy(s2)
    #     pred = cos(s1, s2)>= 0.7
    #     if pred==1 and label==1:
    #         mean_tf[0] += 1
    #     elif pred==1 and label==0:
    #         mean_tf[1] += 1
    #     elif pred==0 and label==1:
    #         mean_tf[2] += 1
    #     else:
    #         mean_tf[3] += 1
    # precision, recall, f1 = evaluate_metrix(mean_tf)
    # print(f"Mean Embedding Precision: {precision}, Recall: {recall}, F1: {f1}")
    
    # merge_tf = [0]*4
    # for item in data.data_pairs_merge:
    #     s1,s2,label = item
    #     s1 = torch.from_numpy(s1).squeeze(0)
    #     s2 = torch.from_numpy(s2).squeeze(0)
    #     pred = cos(s1, s2)>= 0.7
    #     if pred==1 and label==1:
    #         merge_tf[0] += 1
    #     elif pred==1 and label==0:
    #         merge_tf[1] += 1
    #     elif pred==0 and label==1:
    #         merge_tf[2] += 1
    #     else:
    #         merge_tf[3] += 1
    # precision, recall, f1 = evaluate_metrix(merge_tf)
    # print(f"Merge Embedding Precision: {precision}, Recall: {recall}, F1: {f1}")