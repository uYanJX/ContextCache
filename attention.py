import torch
from torch import nn
from torch.nn import functional as F
import math

def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange(maxlen)[None, :] < valid_len[:, None]
    X =X.masked_fill(~mask, value)
    return X

def masker_softmax(x, valid_len):
    if valid_len is None:
        return F.softmax(x,dim=-1)
    else:
        shape = x.shape
        if valid_len.dim()==1:
            valid_len = torch.repeat_interleave(valid_len, repeats=x.size(1))
        else:
            valid_len = valid_len.reshape(-1)
        x = sequence_mask(x.reshape(-1,shape[-1]), valid_len,value=float("-inf"))
        return F.softmax(x.reshape(shape), dim=-1)
    
class DotproductAttention(nn.Module):
    def __init__(self,dropout, **kwargs):
        super(DotproductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
    def forward(self,queries,keys,values,valid_len):
        scores = torch.matmul(queries,keys.transpose(1,2))/math.sqrt(queries.shape[-1])
        attention_weights = masker_softmax(scores, valid_len)
        # print(attention_weights)
        return torch.matmul(self.dropout(attention_weights), values)

def transpose_qkv(X,num_heads):
    X = X.reshape(X.shape[0],X.shape[1],num_heads,-1)
    X = X.permute(0,2,1,3)
    return X.reshape(-1,X.shape[2],X.shape[3])
    
class MultiheadAttention(nn.Module):
    def __init__(self,num_heads,query_size,key_size,value_size,num_hiddens,dropout,**kwargs):
        super(MultiheadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads 
        self.attention = DotproductAttention(dropout)
        self.w_q = nn.Linear(query_size,num_hiddens,bias=False)
        self.w_k = nn.Linear(key_size,num_hiddens,bias=False)
        self.w_v = nn.Linear(value_size,num_hiddens,bias=False)
        self.w_o = nn.Linear(num_hiddens,num_hiddens,bias=False)
    
    def forward(self,queries,keys,values,valid_len):
        if valid_len is not None:
            valid_len = torch.repeat_interleave(valid_len,repeats=self.num_heads,dim=0)
        queries = transpose_qkv(self.w_q(queries),self.num_heads)
        keys = transpose_qkv(self.w_k(keys),self.num_heads)
        values = transpose_qkv(self.w_v(values),self.num_heads)
        output = self.attention(queries,keys,values,valid_len)
        output = output.reshape(-1,self.num_heads,output.shape[1],output.shape[2])
        output = output.permute(0,2,1,3)
        return self.w_o(output.reshape(output.shape[0],output.shape[1],-1))