import torch
import torch.nn as nn 

from layer.feed_forward import FC_FeedForwardnetwork
from layer.layernorm import LayerNorm
from layer.multihead_attention import MultiheadAttention

class DecoderBlock(nn.Module): 
    def __init__(self, d_hidden, num_heads, d_model, drop_rate=0.2): 
        super(DecoderBlock, self).__init__()

        self.d_model = d_model 
        self.d_hidden = d_hidden 
        self.num_heads = num_heads 
        self.drop_rate = drop_rate

        self.attention1 = MultiheadAttention(self.num_heads, self.d_model)
        self.norm1 = LayerNorm(self.d_model)

        self.attention2 = MultiheadAttention(self.num_heads, self.d_model)
        self.norm2 = LayerNorm(self.d_model)

        self.fc_feedforward = FC_FeedForwardnetwork(self.d_model, self.d_hidden, self.drop_rate)
        self.norm3 = LayerNorm(self.d_model)


    def forward(self, enc_src, src_input, src_mask=None): 
        
        X = self.attention1(Q=src_input, K=src_input, V=src_input, mask=src_mask)
        X = self.norm1(X + src_input)

        temp = X.clone().detach()

        X = self.attention2(Q=enc_src, K=enc_src, V=X)
        X = self.norm2(X + temp)

        temp = X.clone().detach()
        X = self.fc_feedforward(X)
        X = self.norm3(X + temp)

        return X 
