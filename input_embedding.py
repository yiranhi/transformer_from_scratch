import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    '''
    input ids (position in the vocabulary) ==> Embedding (vector of 512)
            CAT(6578)                      ==> [2.3, 540.4, ..., 2764.3]
    '''
    def __init__(self, num_embedings, embedding_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embedings, embedding_dim))

    def forward(self, input):
        '''
        input: a sequence of token indices with shape [batch_size, seq_len]
        '''
        return self.weight[input]
    
class tokenizer():
    pass
