"""
The full source code will be made available upon publication of the paper.
Zhao, W., Zhang, B., Zhou, H. et al. Multi-Scale Convolutional Transformer Network for Motor Imagery Brain-Computer Interface. Sci Rep 15, 96611 (2025). doi: 10.1038/s41598-025-96611-5
@author: Wei Zhao
"""

    
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
    def forward(self, x): 
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
