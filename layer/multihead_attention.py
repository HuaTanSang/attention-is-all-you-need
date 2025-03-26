import torch 
import math 
import torch.nn as nn 
from scaled_dot_product_attention import ScaledDotProduct


class MultiheadAttention(nn.Module): 
    def __init__(self, num_heads: int = 8, d_model: int = 512) -> None: 
        super.__init__(MultiheadAttention, self)
        assert (self.d_k * num_heads == d_model), "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_model = d_model 
        
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)
        
        self.attention = ScaledDotProduct(0, self.d_k)
        self.last_linear = nn.Linear(self.d_model, self.d_model)


    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """ 
        Q, K, V: [batch_size, num_heads, num_seq, d_model]
        """
        self.batch_size = Q.size(0) 

        Q = self.w_q(Q) 
        K = self.w_k(K) 
        V = self.w_v(V) 

        mask = mask.unsqueeze(1) # batch_size, num_seq, num_seq 

        Q = self.split_tensor(Q)
        K = self.split_tensor(K) 
        V = self.split_tensor(V) 

        scores = self.attention(Q, K, V, mask)
        scores = scores.transpose(1, 2).contiguous().view(self.batch_size, -1, self.num_heads * self.d_k)
        
        attention = self.last_linear(scores)
        
        return attention
    
    def split_tensor(self, tensor: torch.Tensor):
        """
        Q, V, K need to be divided into num_heads parts
        """
        batch_size = tensor.size(0) 
        tensor = tensor.view(batch_size, -1, self.num_heads, self.d_k) # batch_size, num_seq, num_heads, d_k
        tensor = tensor.transpose(1, 2)    # batch_size, num_heads, num_seq, d_k

        return tensor