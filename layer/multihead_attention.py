import torch 
import torch.nn as nn 
from scaled_dot_product_attention import ScaledDotProductAttention


class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention()
        self.last_linear = nn.Linear(d_model, d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = Q.size(0)
        
        Q = self.w_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)
            
        scores = self.attention(Q, K, V, mask)
        
        scores = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.last_linear(scores)
        
        return output