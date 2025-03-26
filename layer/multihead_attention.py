import torch 
import math 
import torch.nn as nn 


class MultiheadAttention(nn.Module): 
    def __init__(self, num_heads: int = 8, d_model: int = 512) -> None: 
        super.__init__(MultiheadAttention, self)
        self.num_heads = num_heads
        self.d_model = d_model 

        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)

        self.last_linear = nn.Linear(self.d_model, self.d_model)
        self.d_k = d_model // num_heads

        assert (self.d_k * num_heads == d_model), "d_model must be divisible by num_heads"


    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """ 
        Q, K, V: [batch_size, num_heads, num_seq, d_model]
        """

        Q = self.w_q(Q) 
        K = self.w_k(K) 
        V = self.w_v(V) 

        Q, V, K = self.split_tensor(Q, V, K)



        return

    def split_tensor(self, tensor: torch.Tensor):
        """
        Q, V, K need to be divided by num_heads 
        """


        return None 

