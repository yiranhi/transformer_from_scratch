import torch
import torch.nn as nn
import math

class PositionEncoding(nn.Module):
    def __init__(self, d_model:int=512, n:int=10000, max_len:int=5000):
        '''
        Inputs:
            d_model: hidden dimension of the input
            n: user-defined scalar
            max_len: the maximum length of a sequence
        '''
        super().__init__()
        pos_embed = torch.zeros(max_len, d_model)   # [max_len, d_model]

        # position = torch.range(0, max_len, dtype = torch.float).unsqueeze(1) # [max_len, 1]
        '''
        rectify:
            torch.range(start, end) includes the endpoint, i.e. 0~5000, 5001 items in total
            torch.arange(start, end) excludes the endpoint, i.e. 0~4999, 5000 items in total
        '''
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)   #[max_len, 1]

        # div_term = torch.pow(n, (-2 * (max_len // 2)) / d_model)
        '''
        rectify:
            div_term should also be a tensor with shape [1, d_model]. The current defination of
            "div_term" is a single scalar number.
        '''
        div_term = torch.exp(torch.arange(0, d_model, step=2).float() * (- math.log(n) / d_model))  # [d_model // 2]

        pos_embed[:, 0::2] = torch.sin(position * div_term) 
        pos_embed[:, 1::2] = torch.cos(position * div_term)

        pos_embed = pos_embed.unsqueeze(0) # [1, max_len, d_model]

        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def forward(self, x:torch.Tensor):
        '''
        x: the input token sequence with shape [B, seq_len, d_model]
        '''

        return x + self.pos_embed[:, :x.shape[1], :]
