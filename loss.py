import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CosContrastiveLoss(nn.Module):
    def __init__(self, margin=0.7, precision_weight=0.7):
        super().__init__()
        self.margin = margin
        self.precision_weight = precision_weight  # 增加对精确率的重视
        self.cos_sim = nn.CosineSimilarity(dim=-1)  # 计算余弦相似度
        
    def forward(self, anchor, positive=None, negative=None):
        loss = 0
        if positive is not None:
            # 使用余弦相似度
            pos_similarity = self.cos_sim(anchor, positive)
            pos_loss = (1 - self.precision_weight) * (1 - pos_similarity).mean()
            loss += pos_loss
            
        if negative is not None:
            # 使用余弦相似度
            neg_similarity = self.cos_sim(anchor, negative)
            neg_loss = self.precision_weight * torch.clamp(
                self.margin - neg_similarity, min=0
            ).pow(2).mean()
            loss += neg_loss
        
        return loss

# 定义对比损失函数
class BalancedContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0, precision_weight=0.7):
        super().__init__()
        self.margin = margin
        self.precision_weight = precision_weight  # 增加对精确率的重视
        
    def forward(self, anchor, positive=None, negative=None):
        loss = 0
        if positive is not None:
            # 降低正样本权重
            pos_distance = F.pairwise_distance(anchor, positive)
            pos_loss = (1 - self.precision_weight) * pos_distance.pow(2).mean()
            loss += pos_loss
            
        if negative is not None:
            # 增加负样本权重
            neg_distance = F.pairwise_distance(anchor, negative)
            neg_loss = self.precision_weight * torch.clamp(
                self.margin - neg_distance, min=0
            ).pow(2).mean()
            loss += neg_loss

        return loss

class SimpleContrastiveLoss(nn.Module):
    def __init__(self, margin=3.0, precision_weight=0.7, neg_scale=0.5):
        super().__init__()
        self.margin = margin
        self.precision_weight = precision_weight
        self.neg_scale = neg_scale  # 控制难负样本权重的缩放因子
    
    def forward(self, anchor, positive=None, negative=None):
        loss = 0
        
        if positive is not None:
            # 正样本loss保持简单
            pos_distance = F.pairwise_distance(anchor, positive)
            pos_loss = pos_distance.pow(2).mean()
            loss += (1 - self.precision_weight) * pos_loss
        
        if negative is not None:
            # 计算负样本距离
            neg_distance = F.pairwise_distance(anchor, negative)
            # 只对margin范围内的负样本增加权重
            hard_neg_mask = (neg_distance < self.margin).float()
            neg_weight = 1 + self.neg_scale * hard_neg_mask
            
            neg_loss = (neg_weight * torch.clamp(
                self.margin - neg_distance, min=0
            ).pow(2)).mean()
            loss += self.precision_weight * neg_loss
        
        return loss