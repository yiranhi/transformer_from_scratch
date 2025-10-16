import torch
import torch.nn as nn

'''
input matrix [sequence, model_dimension]
    sequence: the length of input sequence
    model_dimension: the word in the input sequemce is transferred into a vector,
                    and the dimension of the vector is defined by the model, is 'model_dimension'
    e.g. we have a sequence composed of 6 words, and each word has been transferred into a 512 dimension vector.
'''

class InputEmbedding(nn.Module):
    '''
    original sentence (tokens) ==> input ids (position in the vocabulary) ==> Embedding (vector of 512)
            CAT                ==>              6578                      ==> [2.3, 540.4, ..., 2764.3]
    '''
    def __init__(self, ):
        self.emb = nn.Embedding()

