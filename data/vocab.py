import torch 
import torch.nn as nn 

from torch.utils.data import Dataset

class Vocab(Dataset): 
    def __init__(self, file_dir):
        self.file_dir = file_dir 
        self.unk_idx = 0 # index for unknown vocab 
    
    def __len__(self): 
        return None 

    def __getitem__(self, index):
        return super().__getitem__(index)
