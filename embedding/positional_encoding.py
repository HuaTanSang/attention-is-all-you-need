import math 
import torch 
import torch.nn as nn 


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        """
        Perform Positional Encoding of Transformer (Encoding base on position of a word in a sentence)
        Arguments: 
            - d_model: dimension of embedding layer. 
            - max_seq_length: the maximum sequence length. 
        Key Features: 
            - Adds positional information to word embeddings so the model can understand the order of tokens in a sequence.

        """
        super(PositionalEncoding, self).__init__()
        
        # zero maxtrix which is correspond to embedding size of a sentence. 
        pe = torch.zeros(max_seq_length, d_model)

        # Initilize positional and div_term coeficient 
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        # Compute additional embedding information
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Save this embedding along the model (when call state_dict method but do not like trainable params)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """ 
        Arguments: 
            - x: Embedding of a sequence. 
        """
        return x + self.pe[:, :x.size(1)]
    