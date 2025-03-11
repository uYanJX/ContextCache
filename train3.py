import os
import torch
import logging
import argparse
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader

# 导入模型和数据集
from model import DialogueSimilarityModel
from dataset2 import DialogueSimilarityDataset, create_dataloader


class DialogueTrainer:
    """
    精简版对话相似度模型训练器
    """
    def __init__(self, train_data_path, val_data_path, gpu_id=0):
        # 基本配置
        self.config = {
            'embed_dim': 768,
            'label_smoothing': 0.05,
            'num_heads': 4,
            'num_layers': 2,
            'dropout_rate': 0.15,
            'temperature': 0.3,
            'similarity_threshold': 0.5,
            'batch_size': 256,
            'eval_batch_size': 64,
            'num_epochs': 30,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'train_file_path': train_data_path,
            'val_file_path': val_data_path,
            'max_seq_length': 10,
            'n_candidates': 10,
            'sentence_model': 'all-mpnet-base-v2',
            'cache_dir': '/data/home/Jianxin/MyProject/ContextCache/cache/test',
            'gpu_id': gpu_id,
            'seed': 42
        }
        
        # 设置随机种子
        self.set_seed(self.config['seed'])
        
        # 设置设备
        self.device = torch.device(f"cuda:{self.config['gpu_id']}" if torch.cuda.is_available() else "cpu")
        
        # 设置实验目录和日志
        self.setup_experiment()
        
    def set_seed(self, seed):
        """设置随机种子以确保可复现性"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        
    def setup_experiment(self):
        """设置实验目录和日志"""
        # 创建实验目录
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(f"results/exp_{time_str}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        
        # 记录基本信息
        self.logger.info(f"实验目录: {self.exp_dir}")
        self.logger.info(f"设备: {self.device}")
    
    def setup_logging(self):
        """配置日志到文件和控制台"""
        log_file = self.exp_dir / "training.log"
        
        # 配置根记录器
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # 获取此类的记录器
        self.logger = logging.getLogger(__name__)
        
    def save_checkpoint(self, model, optimizer, scheduler, epoch, loss, is_best=False):
        """保存模型检查点"""
        # 准备检查点数据
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'config': self.config
        }
        
        # 保存常规检查点
        if (epoch + 1) % 10 == 0:  # 每10个epoch保存一次
            checkpoint_path = self.exp_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"检查点已保存: {checkpoint_path}")
        
        # 如果指示，保存最佳模型
        if is_best:
            best_model_path = self.exp_dir / "best_model.pth"
            torch.save(checkpoint, best_model_path)
            self.logger.info(f"最佳模型已保存: {best_model_path}")
    
    def prepare_data(self):
        """准备数据集和数据加载器"""
        # 初始化数据集
        self.logger.info("初始化训练数据集...")
        train_dataset = DialogueSimilarityDataset(
            data_file_path=self.config['train_file_path'],
            max_seq_length=self.config['max_seq_length'],
            n_candidates=self.config['n_candidates'],
            sentence_model_name=self.config['sentence_model'],
            batch_size=self.config['batch_size'],
            cache_dir=self.config['cache_dir'],
            log=self.logger,
            device=self.device,
            seed=self.config['seed']
        )
        
        self.logger.info("初始化验证数据集...")
        val_dataset = DialogueSimilarityDataset(
            data_file_path=self.config['val_file_path'],
            max_seq_length=self.config['max_seq_length'],
            n_candidates=self.config['n_candidates'],
            sentence_model_name=self.config['sentence_model'],
            batch_size=self.config['batch_size'],
            cache_dir=self.config['cache_dir'],
            log=self.logger,
            device=self.device,
            seed=self.config['seed']
        )
        
        self.logger.info(f"数据集准备完成: {len(train_dataset)} 个训练样本, {len(val_dataset)} 个验证样本")
        
        # 创建数据加载器
        train_loader = create_dataloader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=4
        )
        
        val_loader = create_dataloader(
            val_dataset, 
            batch_size=self.config['eval_batch_size'], 
            shuffle=False,
            num_workers=4
        )
        
        return train_loader, val_loader
    
    def label_smoothing_loss(self, logits, target, smoothing=0.1):
        """计算带有标签平滑的损失"""
        batch_size, n_classes = logits.size()
        
        # 创建one-hot标签
        target_one_hot = torch.zeros(batch_size, n_classes, device=logits.device)
        
        # 处理不匹配情况 (target == -1)
        match_mask = (target >= 0)
        no_match_mask = ~match_mask
        
        # 对于匹配的情况，设置对应的正样本位置
        if match_mask.any():
            match_indices = target[match_mask]
            # 为匹配的样本添加1，因为索引0保留给"无匹配"位置
            adjusted_indices = match_indices + 1
            target_one_hot[match_mask, adjusted_indices] = 1.0
        
        # 对于无匹配的情况，设置索引0为正样本
        if no_match_mask.any():
            target_one_hot[no_match_mask, 0] = 1.0
        
        # 应用标签平滑
        smooth_target = (1.0 - smoothing) * target_one_hot + smoothing / n_classes
        
        # 计算对数概率
        log_probs = F.log_softmax(logits, dim=1)
        
        # 计算损失
        loss = -(smooth_target * log_probs).sum(dim=1).mean()
        
        return loss

    def train(self):
        """训练模型"""
        # 准备数据
        train_loader, val_loader = self.prepare_data()
        
        # 初始化模型
        self.logger.info("初始化模型...")
        model = DialogueSimilarityModel(
            embed_dim=self.config['embed_dim'],
            num_heads=self.config['num_heads'],
            num_layers=self.config['num_layers'],
            dropout_rate=self.config['dropout_rate'],
            temperature=self.config['temperature'],
            similarity_threshold=self.config['similarity_threshold']
        ).to(self.device)
        
        # 初始化优化器
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # 训练循环变量
        best_val_loss = float('inf')
        patience, patience_counter = 8, 0
        
        # 训练循环
        for epoch in range(self.config['num_epochs']):
            # 训练阶段
            train_loss, train_acc = self._run_epoch(model, train_loader, optimizer, is_training=True)
            self.logger.info(f"Epoch {epoch+1}/{self.config['num_epochs']}, "
                            f"训练损失: {train_loss:.6f}, 训练准确率: {train_acc:.4f}")
            
            # 验证阶段
            val_loss, val_acc = self._run_epoch(model, val_loader, is_training=False)
            self.logger.info(f"Epoch {epoch+1}/{self.config['num_epochs']}, "
                            f"验证损失: {val_loss:.6f}, 验证准确率: {val_acc:.4f}")
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 检查这是否是最佳模型
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                self.logger.info(f"新的最佳验证损失: {best_val_loss:.6f}")
            else:
                patience_counter += 1
                
            # 保存检查点
            self.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=val_loss,
                is_best=is_best
            )
            
            # 早停
            if patience_counter >= patience:
                self.logger.info(f"早停在 {epoch+1} 个epoch后触发")
                break
            
        self.logger.info("训练完成!")
        self.logger.info(f"最佳验证损失: {best_val_loss:.6f}")
        
        return model
    
    def _run_epoch(self, model, data_loader, optimizer=None, is_training=True):
        """运行单个epoch的训练或评估"""
        # 设置模型模式
        if is_training:
            model.train()
        else:
            model.eval()
        
        epoch_loss = 0.0
        batch_count = 0
        
        # 额外的指标
        total_samples = 0
        correct_predictions = 0
        
        # 创建进度条
        desc = "训练中" if is_training else "验证中"
        progress_bar = tqdm(data_loader, desc=desc)
        
        # 上下文管理器用于梯度
        context = torch.enable_grad() if is_training else torch.no_grad()
        
        with context:
            for dialogues, masks, labels in progress_bar:
                # 移至设备
                dialogues = dialogues.to(self.device)
                masks = masks.to(self.device) if masks is not None else None
                labels = labels.to(self.device)
                
                # 前向传播
                similarities = model(dialogues, masks)
                
                # 计算损失
                # 应用温度缩放
                logits = similarities / self.config['temperature']
                
                # 训练时使用标签平滑
                if is_training:
                    loss = self.label_smoothing_loss(logits, labels, smoothing=self.config['label_smoothing'])
                else:
                    # 验证时计算标准损失
                    loss = model.compute_loss(similarities, labels)
                
                # 反向传播和优化（如果是训练）
                if is_training and optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    # 梯度裁剪以防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                
                # 更新指标
                batch_loss = loss.item()
                epoch_loss += batch_loss
                batch_count += 1
                
                # 更新准确率指标
                batch_size = dialogues.size(0)
                total_samples += batch_size
                
                # 获取预测（最高相似度）
                predictions = torch.argmax(similarities, dim=1)  # [b]
                
                # 分离无匹配和有匹配的情况
                no_match_mask = (labels == -1)
                match_mask = ~no_match_mask
                
                # 计算总体准确率
                correct_predictions += (
                    (predictions[no_match_mask] == 0).sum().item() +  # 正确的无匹配预测
                    (predictions[match_mask] == (labels[match_mask] + 1)).sum().item()  # 正确的有匹配预测
                )
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f'{batch_loss:.6f}',
                    'acc': f'{correct_predictions/total_samples:.4f}' if total_samples > 0 else 'N/A'
                })
        
        # 计算epoch级别的指标
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        return avg_epoch_loss, accuracy
    
    def evaluate(self, model_path=None):
        """评估训练好的模型"""
        # 准备数据
        _, val_loader = self.prepare_data()
        
        # 初始化模型
        model = DialogueSimilarityModel(
            embed_dim=self.config['embed_dim'],
            num_heads=self.config['num_heads'],
            num_layers=self.config['num_layers'],
            dropout_rate=0.0,  # 评估时不需要dropout
            temperature=self.config['temperature'],
            similarity_threshold=self.config['similarity_threshold']
        ).to(self.device)
        
        # 加载模型检查点
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"从 {model_path} 加载模型")
        else:
            # 尝试加载最佳模型
            best_model_path = self.exp_dir / "best_model.pth"
            if best_model_path.exists():
                checkpoint = torch.load(best_model_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info(f"从 {best_model_path} 加载最佳模型")
            else:
                self.logger.warning("找不到用于评估的模型")
                return None
        
        # 评估
        model.eval()
        with torch.no_grad():
            val_loss, val_acc = self._run_epoch(model, val_loader, is_training=False)
        
        self.logger.info(f"评估完成。损失: {val_loss:.6f}, 准确率: {val_acc:.4f}")
        
        return {'val_loss': val_loss, 'accuracy': val_acc}


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="训练对话相似度模型")
    parser.add_argument("--train_data_path", type=str, 
                        default="/data/home/Jianxin/MyProject/ContextCache/data/new/train.jsonl")
    parser.add_argument("--val_data_path", type=str, 
                        default="/data/home/Jianxin/MyProject/ContextCache/data/new/val.jsonl")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"], default="train")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=30)
    
    args = parser.parse_args()
    
    # 初始化训练器
    trainer = DialogueTrainer(
        train_data_path=args.train_data_path, 
        val_data_path=args.val_data_path, 
        gpu_id=args.gpu_id
    )
    
    # 用命令行参数更新配置
    trainer.config.update({
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs
    })
    
    if args.mode == "evaluate":
        # 评估模型
        results = trainer.evaluate(model_path=args.model_path)
        if results:
            print(f"评估结果: 损失={results['val_loss']:.6f}, 准确率={results['accuracy']:.4f}")
    else:
        # 训练模型
        model = trainer.train()