import torch 
import torch.nn as nn 


class TokenEmbedding(nn.Module): 
    def __init__(self, num_seq, embedding_dim): 
        super(TokenEmbedding, self).__init__()
        
        self.num_seq = num_seq
        self.embedding_dim = embedding_dim

    def __getitem__(self): 
        return None 