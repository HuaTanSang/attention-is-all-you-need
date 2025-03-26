import torch.nn as nn 

class FC_FeedForwardnetwork(nn.Module): 
    def __init__(self, d_model, d_hidden): 
        super(FC_FeedForwardnetwork, self).__init__() 
        self.d_model = d_model 
        self.d_hidden = d_hidden

        self.first_linear = nn.Linear(self.d_model, self.d_hidden)
        self.second_linear = nn.Linear(self.d_hidden, self.d_model) 

        self.relu = nn.ReLU()

    def forward(self, X): 
        output = self.first_linear(X) 
        output = self.relu(output)
        output = self.second_linear(output) 

        return output 