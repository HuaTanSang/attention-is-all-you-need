import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class PositionalEncoding(nn.Module): 
    def __init__(self):
        super(PositionalEncoding, self).__init__() 


    def forward(self, embedding):
        batch_size, num_seq, embedding_dim = embedding.size() 

        pe = torch.zeros(batch_size, num_seq, embedding_dim)

        for batch in batch_size:
            for position in range(0, num_seq, 2): 
                pe[batch][position] = math.sin(position / (10000**(2*position/embedding)))
                pe[batch][position+1] = math.cos((position+1) / (10000**(2*(position+1)/embedding_dim)))
    
        return pe + embedding
    
            