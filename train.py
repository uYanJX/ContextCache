import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from dataset import QueryDataset
from datetime import datetime
import logging
from pathlib import Path
import yaml
from pathlib import Path
import argparse
import random
import json

# 定义模型类
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from loss import *
    
def generate_padding_mask(batch, pad_value=0):
    """Key Padding Mask, where elem to ignore"""
    return (batch.sum(dim=-1) == pad_value)

def evaluate_metrix(records):
    precision = records[0] / (records[0] + records[1]) if (records[0] + records[1]) > 0 else 0
    recall = records[0] / (records[0] + records[2]) if (records[0] + records[2]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

class ContextTrainer:
    def __init__(self, config_path=None, margin=None, gpu_id=None, num_layers=None):
        self.config = self._load_config(config_path) if config_path else self._default_config()
        self.config['margin'] = margin if margin else self.config['margin']
        self.config['gpu_id'] = gpu_id if gpu_id else self.config['gpu_id']
        self.config['num_layers'] = num_layers if num_layers else self.config['num_layers']
        self.device = torch.device(f"cuda:{self.config['gpu_id']}" if torch.cuda.is_available() else "cpu")
        self.setup_experiment()
        self.first_epoch = True
        
    def _default_config(self):
        return {
            'embed_dim': 768,
            'num_heads': 8,
            'num_layers': 2,
            'batch_size': 512,
            'num_epochs': 30,
            'learning_rate': 1e-4,
            'margin': 3.0,
            'pos_path': '/data/home/Jianxin/MyProject/ContextCache/data/final/pos_dataset.jsonl',
            'neg_path': '/data/home/Jianxin/MyProject/ContextCache/data/final/neg_dataset.jsonl',
            'gpu_id': 5 
        }
    
    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_experiment(self):
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(f"results/exp_{time_str}_{self.config['gpu_id']}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
        self.save_config()
        
    def setup_logging(self):
        log_file = self.exp_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def save_config(self):
        config_path = self.exp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
            
    def save_checkpoint(self, model, optimizer, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        checkpoint_path = self.exp_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
    def load_checkpoint(self, model, optimizer, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']

    def train(self, resume_from=None):
        # 初始化模型、损失函数和优化器
        model = OptimizedMHA5(
            embed_dim=self.config['embed_dim'],
            num_heads=self.config['num_heads'],
            # num_layers=self.config['num_layers']
        )
        model = model.to(self.device)
        criterion = SimpleContrastiveLoss(margin=self.config['margin'])
        optimizer = optim.AdamW(model.parameters(), lr=self.config['learning_rate'])# , weight_decay=1e-4)
        
        # 加载数据集
        dataset = QueryDataset(
            pos_file_path=self.config['pos_path'],
            neg_file_path=self.config['neg_path'],
            device=self.device, 
            batch_size=512,
            log = self.logger,
        )
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        # 恢复检查点（如果有）
        start_epoch = 0
        if resume_from:
            start_epoch, loss = self.load_checkpoint(model, optimizer, resume_from)
            self.logger.info(f"Resumed from epoch {start_epoch} with loss {loss}")

        # 训练循环
        best_val_loss = float('inf')
        for epoch in range(start_epoch, self.config['num_epochs']):
            # 训练阶段
            train_loss = self._run_epoch(model, train_loader, epoch, criterion, optimizer, is_training=True)
            self.logger.info(f"Epoch {epoch+1}/{self.config['num_epochs']}, Train Loss: {train_loss:.4f}")
            # 验证阶段
            val_loss = self._run_epoch(model, val_loader, epoch, criterion, is_training=False)
            self.logger.info(f"Epoch {epoch+1}/{self.config['num_epochs']}, Val Loss: {val_loss:.4f}")            
            # 保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(model, optimizer, epoch+1, val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = self.exp_dir / "best_model.pth"
                torch.save(model.state_dict(), best_model_path)
                # self.logger.info(f"Best model saved with validation loss: {best_val_loss:.4f}")

    def get_TFinfo(self, s1_repr, s2_repr, measure, threshold, matrix, label):
        cosine_sim = measure(s1_repr, s2_repr)
        pred_labels = (cosine_sim >= threshold).float()
        matrix[0] += ((pred_labels == 1) & (label == 1)).sum().item()
        matrix[1] += ((pred_labels == 1) & (label == 0)).sum().item()
        matrix[2] += ((pred_labels == 0) & (label == 1)).sum().item()
        matrix[3] += ((pred_labels == 0) & (label == 0)).sum().item()

    def _run_epoch(self, model, data_loader, epoch ,criterion, optimizer=None, is_training=True):
        model.train() if is_training else model.eval()
        epoch_loss = 0.0

        desc = f"Epoch {epoch+1}/{self.config['num_epochs']} [{'Train' if is_training else 'Validation'}]"
        context = torch.enable_grad() if is_training else torch.no_grad()
        
        # true positives, false positives, false negatives, true negatives
        matrix_TF = [ 0 ] * 4
        cos = nn.CosineSimilarity(dim=1)
        my_threshold = 0.7
        
        with context:
            for batch in tqdm(data_loader, desc=desc):
                s1, s2, label = batch
                pos_mask = label == 1
                neg_mask = label == 0
                
                s1_mask = generate_padding_mask(s1)
                s2_mask = generate_padding_mask(s2)
                
                anchor_repr = model(s1, key_padding_mask=s1_mask)
                posneg_repr = model(s2, key_padding_mask=s2_mask)
                self.get_TFinfo(anchor_repr, posneg_repr, cos, my_threshold, matrix_TF, label)
        
                pos_loss = criterion(anchor_repr[pos_mask], posneg_repr[pos_mask],None) if pos_mask.any() else 0
                neg_loss = criterion(anchor_repr[neg_mask], None, posneg_repr[neg_mask]) if neg_mask.any() else 0
                loss = pos_loss + neg_loss
                
                # if not is_training and self.first_epoch:
                #     self.first_epoch = False
                #     s1_mean = (s1 * ~s1_mask.unsqueeze(-1)).sum(dim=1) / (~s1_mask).sum(dim=1, keepdim=True)
                #     s2_mean = (s2 * ~s2_mask.unsqueeze(-1)).sum(dim=1) / (~s2_mask).sum(dim=1, keepdim=True)
                #     self.get_TFinfo(s1_mean, s2_mean, cos, mean_threshold, matrix_TF_mean, label)
                
                if is_training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                epoch_loss += loss.item()
                
        avg_loss = epoch_loss / len(data_loader)

        # if not is_training:
        # self.logger.info(f"{'Train' if is_training else 'Val'} Loss - MHA: {avg_loss:.4f}")        
        precision, recall, f1 = evaluate_metrix(matrix_TF)
        # baseline_precision, baseline_recall, baseline_f1 = evaluate_metrix(matrix_TF_mean)
        
        self.logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        self.logger.info(f"True Positives: {matrix_TF[0]}, False Positives: {matrix_TF[1]}, False Negatives: {matrix_TF[2]}, True Neg: {matrix_TF[3]}")
        # self.logger.info(f"Baseline Metrics:")
        # self.logger.info(f"Precision: {baseline_precision:.4f}, Recall: {baseline_recall:.4f}, F1 Score: {baseline_f1:.4f}")
        # self.logger.info(f"True Positives: {matrix_TF_mean[0]}, False Positives: {matrix_TF_mean[1]}, False Negatives: {matrix_TF_mean[2]}, True Neg: {matrix_TF_mean[3]}")
            
        return avg_loss

if __name__ == "__main__":
    # 添加参数解析
    parser = argparse.ArgumentParser()
    # albert margin 2-2.5
    parser.add_argument("--margin", type=float, default=3, help="Margin for contrastive loss")
    parser.add_argument("--gpu_id", type=int, default=4, help="GPU ID to run the training")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the model")
    args = parser.parse_args()

    # 将参数传递给 ContextTrainer
    trainer = ContextTrainer(margin=args.margin, gpu_id=args.gpu_id, num_layers=args.num_layers)
    trainer.train()


