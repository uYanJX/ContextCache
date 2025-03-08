import os
import re
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from functools import partial
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

class DialogueSimilarityDataset(Dataset):
    def __init__(self, 
                 data_file_path,
                 max_seq_length=10,
                 n_candidates=20,  # 每个查询对应的候选数量（包括1个正样本和n_candidates-1个负样本）
                 sentence_model_name="all-mpnet-base-v2",
                 batch_size=512,
                 cache_dir="/data/home/Jianxin/MyProject/ContextCache/cache/test",
                 log=None,
                 device=None):
        """
        针对BatchDialogueSimilarityModel设计的数据集
        
        Args:
            pos_file_path: 正样本文件路径
            neg_file_path: 负样本文件路径
            max_seq_length: 最大序列长度
            n_candidates: 每个查询对应的候选数量（包括1个正样本和n_candidates-1个负样本）
            sentence_model_name: 句子编码模型名称
            batch_size: 编码批次大小
            cache_dir: 缓存目录
            log: 日志对象
            device: 设备
        """
        self.logger = log
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_seq_length = max_seq_length
        self.n_candidates = n_candidates
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 尝试加载缓存
        cache_file = f"dialog_sim_{os.path.basename(data_file_path).split('.')[0]}.pkl"
        cache_path = os.path.join(cache_dir, cache_file)
        
        # 加载句子编码模型
        self.sentence_model = SentenceTransformer(sentence_model_name).to(self.device)
        
        # 加载数据并预处理
        data = [json.loads(line) for line in open(data_file_path)]
        
        all_sentences, cleaned_data = self._preprocess_data(data)

        # 清理和编码所有句子
        if os.path.exists(cache_path):
            embeddings = torch.load(cache_path)
            if self.logger:
                self.logger.info(f"Loaded data from cache: {cache_path}")
        else:
            embeddings = self._parallel_encode(all_sentences, batch_size)
            # 保存缓存
            torch.save(embeddings, cache_path)
        
        # 构建数据集
        self.data = self._build_dataset(cleaned_data, embeddings)
        if self.logger:
            self.logger.info(f"Saved dataset to cache: {cache_path}")
            self.logger.info(f"Dataset contains {len(self.data)} dialogue groups")

    def _clean_numbering(self, text):
        """清理文本中的编号"""
        sentences = re.split(r'^\d+\.\s|(?:\n)\d+\.\s', text, flags=re.MULTILINE)
        return [re.sub(r'^\d+\.\s*', '', sentence) for sentence in sentences if sentence.strip()]
    
    def _preprocess_data(self, data):
        """预处理数据，清理文本，返回清理后的数据"""
        all_sentences = set()
        
        # 清理正样本数据
        cleaned_data = []
        for item in data:
            cleaned_item = {}
            for key in ["original", "variations", "simplified", "expanded"]:
                if key in item:
                    cleaned_sentences = self._clean_numbering(item[key])
                    cleaned_item[key] = cleaned_sentences
                    all_sentences.update(cleaned_sentences)
            if "neg" in item:
                cleaned_item["neg"] = self._clean_numbering(item["neg"])
                all_sentences.update(cleaned_item["neg"])
            
            if cleaned_item:
                cleaned_data.append(cleaned_item)
        
        return all_sentences, cleaned_data
    
    def _build_dataset(self, cleaned_data , embeddings):
        """
        构建数据集
        
        每个数据项格式为: 
        {
            'dialogues': [query_dialogue, positive_dialogue, negative_dialogue_1, ..., negative_dialogue_n-1],
            'masks': [query_mask, positive_mask, negative_mask_1, ..., negative_mask_n-1] 
        }
        
        其中:
        - dialogues[0]是查询对话，shape为[max_seq_length, dim]
        - dialogues[1]是与查询相关的正样本对话
        - dialogues[2:]是负样本对话
        - masks对应的是padding掩码，True表示padding位置
        """
        dataset = []
        
        origin_data = [item["original"] for item in cleaned_data] 
        
        # 从正样本数据构建样本
        for item in tqdm(cleaned_data, desc="Processing"):
            if "original" not in item:
                continue
                
            query = item["original"]
            query_embeddings = np.array([embeddings[s] for s in query])
            
            # 为每个变体创建一个样本
            for key in ["variations", "simplified", "expanded"]:
                if key not in item or not item[key]:
                    continue
                    
                positive = item[key]
                positive_embeddings = np.array([embeddings[s] for s in positive])
                
                hard_neg = item["neg"]
                # 收集足够的负样本
                negative_samples = self._collect_negative_samples(
                    hard_neg, 
                    embeddings, 
                    self.n_candidates - 1,  # 减1是因为已经有一个正样本
                    query,
                    origin_data
                )
                
                # 组合查询、正样本和负样本
                all_dialogues = [query_embeddings, positive_embeddings]
                all_dialogues.extend(negative_samples)
                
                # 创建掩码
                masks = []
                for dialogue in all_dialogues:
                    mask = np.zeros(self.max_seq_length, dtype=bool)
                    if len(dialogue) < self.max_seq_length:
                        mask[len(dialogue):] = True
                    masks.append(mask)
                
                # 添加到数据集
                dataset.append({
                    'dialogues': [self._pad_sequence(d) for d in all_dialogues],
                    'masks': masks
                })
        
        # 打乱数据集
        random.shuffle(dataset)
        return dataset
    
    def _collect_negative_samples(self, neg_data, embeddings, n_samples, query, data):
        """收集负样本"""
        negative_samples = []
        
        # hard_neg_embeddings = np.array([embeddings[s] for s in neg_data])
        # negative_samples.append(hard_neg_embeddings)
        
        if len(query) > 1:
            query_embeddings = np.array([embeddings[s] for s in query])
            idx = random.randint(0, len(query)-2)
            shuffled = query_embeddings.copy()
            # 随机打乱顺序
            shuffled[idx],shuffled[-1] = shuffled[-1],shuffled[idx]
            np.random.shuffle(shuffled[:-1])
            negative_samples.append(shuffled)
            
        if len(query) > 1:
            query_embeddings = np.array([embeddings[s] for s in query])
            idx = random.randint(1, len(query)-1)
            negative_samples.append(query_embeddings[:idx])
                
        # 从其他样本数据中收集
        left = n_samples - len(negative_samples)
        while left:
            random_neg = random.choice(data)
            if random_neg == query:
                continue
            random_neg_embeddings = np.array([embeddings[s] for s in random_neg])
            negative_samples.append(random_neg_embeddings)
            left -= 1
        
        return negative_samples
    
    def _pad_sequence(self, sequence):
        """对序列进行填充或截断"""
        if len(sequence) > self.max_seq_length:
            return sequence[:self.max_seq_length]
        
        padded_shape = (self.max_seq_length, sequence.shape[1])
        padded = np.zeros(padded_shape)
        padded[:len(sequence)] = sequence
        return padded
    
    def __getitem__(self, idx):
        """
        返回单个数据项
        
        Returns:
            dialogues: 形状为[n, max_seq_length, dim]的张量
            masks: 形状为[n, max_seq_length]的张量
        """
        item = self.data[idx]
        
        # 转换为张量
        dialogues = torch.tensor(np.array(item['dialogues']), dtype=torch.float32)
        masks = torch.tensor(np.array(item['masks']), dtype=torch.bool)
        
        return dialogues, masks
    
    def __len__(self):
        return len(self.data)
    
    def _encode_batch(self, batch_and_gpu):
        """对单个批次进行编码"""
        sentences, gpu_id = batch_and_gpu
        device = torch.device(gpu_id)
        return self.sentence_model.encode(
            sentences,
            convert_to_numpy=True,
            device=device,
            show_progress_bar=False
        )

    def _parallel_encode(self, all_sentences, batch_size, num_workers=6, gpus=None):
        """使用多进程并行对句子进行编码"""
        embeddings = {}
        sentences_list = list(all_sentences)
        total_batches = (len(sentences_list) + batch_size - 1) // batch_size

        # 将数据分成多个批次
        batches = [
            sentences_list[i:i + batch_size]
            for i in range(0, len(sentences_list), batch_size)
        ]

        # 确保有足够的GPU分配
        if gpus is None:
            # 默认使用可用的所有GPU
            num_gpus = torch.cuda.device_count()
            if num_gpus > 0:
                gpus = [f"cuda:{i}" for i in range(num_gpus)]
            else:
                gpus = ["cpu"]
                num_workers = 1
        
        assert len(gpus) >= 1, "At least one GPU or CPU must be available"
        num_workers = min(num_workers, len(gpus))

        # 设置多进程上下文
        ctx = mp.get_context('spawn')

        with ctx.Pool(num_workers) as pool:
            # 为每个批次分配GPU（轮询分配）
            gpu_assignments = [gpus[i % len(gpus)] for i in range(len(batches))]
            tasks = zip(batches, gpu_assignments)

            # 并行处理批次
            results = list(tqdm(
                pool.imap(partial(self._encode_batch), tasks),
                total=total_batches,
                desc="Encoding sentences"
            ))

        # 合并结果
        for batch, encodings in zip(batches, results):
            embeddings.update(dict(zip(batch, encodings)))

        return embeddings


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=4):
    """创建数据加载器"""
    def collate_fn(batch):
        """自定义批处理函数，将多个样本组合成一个批次"""
        dialogues_batch = []
        masks_batch = []
        
        for dialogues, masks in batch:
            dialogues_batch.append(dialogues)
            masks_batch.append(masks)
        
        # 堆叠为形状[b, n, max_seq_length, dim]的张量
        dialogues_tensor = torch.stack(dialogues_batch)
        masks_tensor = torch.stack(masks_batch)
        
        return dialogues_tensor, masks_tensor
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


# 使用示例
def example_usage():
    # 配置参数
    file = "/data/home/Jianxin/MyProject/ContextCache/data/new/train.jsonl"
    max_seq_length = 10
    n_candidates = 10  # 每个查询有1个正样本和4个负样本
    batch_size = 16
    
    # 创建数据集
    dataset = DialogueSimilarityDataset(
        data_file_path=file,
        max_seq_length=max_seq_length,
        n_candidates=n_candidates
    )
    
    # 创建数据加载器
    dataloader = create_dataloader(dataset, batch_size=batch_size)
    
    # 打印数据集信息
    print(f"Dataset size: {len(dataset)}")
    
    # 获取一个批次的数据并检查形状
    for dialogues, masks in dataloader:
        print(f"Batch dialogues shape: {dialogues.shape}")  # 应为[batch_size, n_candidates+1, max_seq_length, embed_dim]
        print(f"Batch masks shape: {masks.shape}")  # 应为[batch_size, n_candidates+1, max_seq_length]
        break
    
if __name__ == "__main__":
    example_usage()