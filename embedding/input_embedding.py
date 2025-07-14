import torch 
import torch.nn as nn 

class InputEmbedding(nn.Module): 
    def __init__(self,
                 vocab_size: int, 
                 embedding_dim: int,
                 padding_idx: int = 0,
                 trainable: bool = True,
                 device: str = "cpu"): 
        """
        Standard embedding layer for Transformer models.
        Arguments: 
            - vocab_size: size of the vocabulary
            - embedding_dim: dimension of embeddings
            - padding_idx: index for padding token
            - trainable: whether to allow fine-tuning embeddings
            - device: torch device
        """
        super(InputEmbedding, self).__init__() 

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        if not trainable:
            self.embedding.weight.requires_grad = False
        self.device = "cuda" if torch.cuda.is_available() else device 

    def forward(self, input_ids: torch.Tensor):
        """
        Arguments:  
            - input_ids: Tensor of shape [batch, seq_len] or [seq_len]
        Returns: 
            - Tensor of shape [batch, seq_len, embedding_dim] or [seq_len, embedding_dim]
        """ 
        return self.embedding(input_ids.to(self.device))

    def __getitem__(self, idx: int): 
        """Extract embedding vector by index"""
        return self.embedding.weight[idx]