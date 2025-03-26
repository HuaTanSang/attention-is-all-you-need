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
            for position in num_seq: 
                if ((position & 1) == 0):
                    pe[batch][position] = math.sin(position / (10000**(2*position/embedding)))
                else: 
                    pe[batch][position] = math.cos(position / (10000**(2*(position+1)/embedding_dim)))
        
        return pe
            