import torch.nn as nn 

from layer.multihead_attention import MultiheadAttention
from layer.feed_forward import FC_FeedForwardnetwork
from layer.layernorm import LayerNorm


class EncoderBlock(nn.Module): 
    def __init__(self, d_hidden=1024, num_heads=8, d_model=512, drop_rate=0.2): 
        super(EncoderBlock, self).__init__() 
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.drop_rate = drop_rate

        self.attention = MultiheadAttention(self.num_heads, self.d_model)
        self.norm1 = LayerNorm(self.d_model)
        self.drop1 = nn.Dropout(p=self.drop_rate)

        self.fc_feedforward = FC_FeedForwardnetwork(self.d_model, self.d_hidden, self.drop_rate)
        self.norm2 = LayerNorm(self.d_model)
        self.drop2 = nn.Dropout(self.drop_rate)

    def forward(self, src_input, src_mask=None):

        X = self.attention(Q=src_input, K=src_input, V=src_input, mask=src_mask)
        X = self.norm1(src_input + X)

        temp = X 
        X = self.fc_feedforward(X)
        X = self.norm2(temp + X) 

        return X
        