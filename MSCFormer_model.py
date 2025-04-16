"""
Zhao, W., Zhang, B., Zhou, H. et al. Multi-scale convolutional transformer network for motor imagery brain-computer interface. Sci Rep 15, 12935 (2025). 
https://doi.org/10.1038/s41598-025-96611-5

@author: Wei Zhao
"""

import numpy as np
import pandas as pd
import random
import datetime
import time
import os
from pandas import ExcelWriter
from torchsummary import summary
import torch
from torch.backends import cudnn
from utils import calMetrics
from utils import calculatePerClass
from utils import numberClassChannel
import math

cudnn.benchmark = False
cudnn.deterministic = True



import torch
from torch import nn
from torch import Tensor
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
import torch.nn.functional as F

from utils import numberClassChannel
from utils import load_data_evaluate

import numpy as np
import pandas as pd
from torch.autograd import Variable


class PatchEmbeddingCNN(nn.Module):
    def __init__(self, 
                 f1=16, 
                 pooling_size=52, 
                 dropout_rate=0.5, 
                 number_channel=22):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 85), (1, 1), padding='same'),
            nn.Conv2d(f1, f1, (number_channel, 1), (1, 1), groups=f1),
            nn.BatchNorm2d(f1),
            nn.ELU(),
            nn.AvgPool2d((1,pooling_size)), 
            nn.Dropout(dropout_rate),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 65), (1, 1), padding='same'),
            nn.Conv2d(f1, f1, (number_channel, 1), (1, 1), groups=f1),
            nn.BatchNorm2d(f1),
            nn.ELU(),
            nn.AvgPool2d((1,pooling_size)), 
            nn.Dropout(dropout_rate),
        )        
        self.cnn3 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 45), (1, 1), padding='same'),
            nn.Conv2d(f1, f1, (number_channel, 1), (1, 1), groups=f1),
            nn.BatchNorm2d(f1),
            nn.ELU(),
            nn.AvgPool2d((1,pooling_size)), 
            nn.Dropout(dropout_rate),
        )
        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x1 = self.cnn1(x)
        x2 = self.cnn2(x)
        x3 = self.cnn3(x)
        #通道方向合并
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.projection(x)
        return x    
    
  
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


# PointWise FFN
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class ClassificationHead(nn.Sequential):
    def __init__(self, flatten_number, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(flatten_number, n_classes)
        )

    def forward(self, x):
        out = self.fc(x)
        
        return out

# add & LayerNorm
class ResidualAdd(nn.Module):
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, **kwargs):
        x_input = x
        res = self.fn(x, **kwargs)
        out = self.layernorm(self.drop(res)+x_input)
        return out

    
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=4,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                MultiHeadAttention(emb_size, num_heads, drop_p),
                ), emb_size, drop_p),
            ResidualAdd(nn.Sequential(
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                ), emb_size, drop_p)
            )    
        
        
class TransformerEncoder(nn.Sequential):
    def __init__(self, heads, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size, heads) for _ in range(depth)])


class BranchEEGNetTransformer(nn.Sequential):
    def __init__(self, parameters,
                 **kwargs):
        super().__init__(
            PatchEmbeddingCNN(f1=parameters.f1, 
                                 pooling_size=parameters.pooling_size, 
                                 dropout_rate=parameters.dropout_rate,
                                 number_channel=parameters.number_channel,
                                 ),
        )


class PositioinalEncoding(nn.Module):
    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))
    def forward(self, x): # x-> [batch, embedding, length]
        x = x + self.encoding[:, :x.shape[1], :].cuda()
        return self.dropout(x)        
        
    
class MSCFormer(nn.Module):
    def __init__(self, 
                 parameters,
                 database_type='A', 
                 
                 **kwargs):
        super().__init__()
        self.number_class, self.number_channel = numberClassChannel(database_type)
        self.emb_size = parameters.emb_size
        parameters.number_channel = self.number_channel
        self.cnn = BranchEEGNetTransformer(parameters)
        self.position = PositioinalEncoding(parameters.emb_size, dropout=0.1)
        self.trans = TransformerEncoder(parameters.heads, 
                                        parameters.depth, 
                                        parameters.emb_size)

        self.flatten = nn.Flatten()
        self.classification = ClassificationHead(self.emb_size , self.number_class) 
    def forward(self, x):
        x = self.cnn(x)
        b, l, e = x.shape
        x = torch.cat((torch.zeros((b, 1, e),requires_grad=True).cuda(),x), 1)
        x = x * math.sqrt(self.emb_size)
        x = self.position(x)
        trans = self.trans(x)
        features = trans[:, 0, :]
        
        out = self.classification(features)
        return features, out
