import torch

class SelfAttention():
    '''
        Note: input token sequence = Q = K = V (Q, K, V are just the copy of input token sequence)    
    
        input token sequence: [sequence, model_dimension]
        Q, K, V: [sequence, model_dimension]
        e.g. consider that we have a sentence composed of 6 words, and each word is converted into
        a 512 dimensions vector.

        softmax(torch.dot(Q [6, 512], K.T [512, 6]) / troch.sqrt(512) ) ==> attention score[6, 6]
        torch.dot(attention score [6, 6], V [6, 512]) ==> attention [6, 512] 
    '''

class MultiHeadAttention():
    '''
        the matrix of W_q, W_k, W_v, W_o: [model_dimension, model_dimension]
        torch.dot(Q [6, 512], W_q [512, 512]) ==> Q' [6, 512]
        torch.dot(K [6, 512], W_k [512, 512]) ==> K' [6, 512]
        torch.dot(V [6, 512], W_v [512, 512]) ==> V' [6, 512]

        divide Q', K', V' along the dimension 'model_dimension', then every head
        can see the whole sequence but part of the word vector.

        e.g. if there are 4 heads
        Q' [6, 512] ==> Q1 [6, 128], Q2 [6, 128], Q3 [6, 128], Q4 [6, 128]
        K' [6, 512] ==> K1 [6, 128], K2 [6, 128], K3 [6, 128], K4 [6, 128]
        V' [6, 512] ==> V1 [6, 128], V2 [6, 128], V3 [6, 128], V4 [6, 128]

        SelfAttention(Q1, K1, V1) ==> head1 [6, 128]
        SelfAttention(Q2, K2, V2) ==> head2 [6, 128]
        SelfAttention(Q3, K3, V3) ==> head3 [6, 128]
        SelfAttention(Q4, K4, V4) ==> head4 [6, 128]

        torch.cat(head1, head2, head3, head4) ==> concatenate_attention [6, 128*4=512]
        torch.dot(concatenate_attention [6, 512], W_o [512, 512]) ==> multi-head attention [6, 512]

    '''

class MaskedMultiHeadAttention():
    '''
        For Causal Model:
            the output at a certain position can only depend on the words on previous positions.
            The model 'MUST NOT' be able to see future words.

        attention socre [sequence, sequence]:
            To implment Causal relationship, we need to set all the values above the diagonal as '-inf'
            before applyling softmax. And those position will become '0' after doing softmax.
              ———— ———— ———— ————
            | 6.1  -inf -inf -inf |
            | 2.9  4.7  -inf -inf |
            | 3.5  2.4  9.2  -inf |
            | 7.8  3.3  8.4   4.9 |
              ____ ____ ____ ____
    '''