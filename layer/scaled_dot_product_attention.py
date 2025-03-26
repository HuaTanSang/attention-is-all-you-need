import math 
import torch 
import torch.nn as nn 

class ScaledDotProduct(nn.Module): 
    def __init__(self,  head: int, d_model): 
        super(ScaledDotProduct, self).__init__()

        self.d_model = d_model
        self.head = head 

        self.w_Q = nn.Linear(self.d_model, self.d_model)
        self.w_K = nn.Linear(self.d_model, self.d_model)
        self.w_V = nn.Linear(self.d_model, self.d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None) -> torch.Tensor: 
        """
        Q, K, V: batch_size, head, seq_len, d_model 
        """

        Q = self.w_Q(Q) 
        K = self.w_K(K) 
        V = self.w_V(V) 

        d_k = K.size(-1) # d_model 

        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(d_k)
        if mask: 
            scores = scores.masked_fill(mask==0, 1e-9)
        scores = self.softmax(scores, dim=-1)
        attention = torch.matmul(scores, V)

        return attention
    








