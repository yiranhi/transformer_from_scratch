import torch
import torch.nn as nn

class PositionWiseFeedForwaed(nn.Module):
    '''
    Position-Wise:
        it means "the same linear transformation is used on each token in sequence independently"
    '''
    def __init__(self, d_model:int=512, d_hidden:int=2048):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))