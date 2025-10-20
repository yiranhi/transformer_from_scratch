import torch
import torch.nn as nn
import math

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

class MultiHeadAttention(nn.Module):
    '''
        model_dimension: the vector(feature) dimenesion of each token
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

    def __init__(self, num_heads:int, d_model:int):
        super().__init__()
        # assert d_model // num_heads == 0
        '''
        Modified Method
        '''
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model // num_heads

        # self.qkv_proj = nn.Linear(d_model, d_model)
        '''
        Error: 
            The q_proj(W_q), k_proj(W_k), v_proj(W_v) shouldn't share the same weights

        More Efficient method to handle this is producing 3 projection layers at one time and chunk it
        into 3 sub-matrix

        e.g.
        qkv_proj = nn.Linear(d_model, 3*d_model, bias = False)
        Q, K, V = qkv_proj(x).chunk(3, dim=-1)

        Smarter Method:
        '''
        self.qkv_proj = nn.Linear(d_model, 3*d_model, bias = False)
        # 'bais = True' or 'bais = False' either works. 'bias = False' save parameters with negligible impact on quality

        
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x:torch.Tensor):
        batch_size, seq_len, d_model = x.shape

        # Q = self.qkv_proj(x)    # [B, seq_len, d_model]
        # K = self.qkv_proj(x)    # [B, seq_len, d_model]
        # V = self.qkv_proj(x)    # [B, seq_len, d_model]
        '''
        Modeified Method
        '''
        Q, K, V = self.qkv_proj(x).chunk(3, dim=-1) # [B, seq_len, d_model]


        # Q_heads = Q.permute(0, 2, 1).view(batch_size, self.num_heads, self.d_head, seq_len).permute(0, 1, 3, 2)   # [B, num_heads, seq_len, d_head]
        # K_heads = K.permute(0, 2, 1).view(batch_size, self.num_heads, self.d_head, seq_len).permute(0, 1, 3, 2)   # [B, num_heads, seq_len, d_head]
        # V_heads = V.permute(0, 2, 1).view(batch_size, self.num_heads, self.d_head, seq_len).permute(0, 1, 3, 2)   # [B, num_heads, seq_len, d_head]
        '''
        Modified Methods:
            Q_heads = Q.permute(0, 2, 1).contiguous().view(batch_size, self.num_heads, self.d_head, seq_len).permute(0, 1, 3, 2)
        Note: remeber to use 'contiguous()' after 'tranpose()'/'permute()' and before 'view()'.
              because 'view()' requires the tensor's layout in memory to be compatible, while 
              'transpose()'/'permute()' will destory the continuity.
        
        Smarter Methods:
        '''
        def to_heads(t):
            return t.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        Q_heads, K_heads, V_heads = map(to_heads, (Q, K, V))    # [B, num_heads, seq_len, d_head]

        
        # attention_logits = torch.matmul(Q_heads, K_heads.T) / torch.sqrt(self.d_head)   # [B, num_heads, seq_len, seq_len]
        '''
        Error:
            1. T is used for 2D tensor only. For high dimension tensor, you should use 'transpose()'
            2. self.d_head is not a tensor, 'torch.sqrt' only works on tensor. 
        
        Modified Method:
        '''
        attention_logits = torch.matmul(Q_heads, K_heads.transpose(-1, -2)) / math.sqrt(self.d_head)    # [B, num_heads, seq_len, seq_len]


        # attention_score = torch.softmax(attention_logits)
        '''
        Error:
            'softmax' needs to identify the dimension to normalize. In this case, the dimesion is -1('kv_len').
        
        Modified Method:
        '''
        attention_score = torch.softmax(attention_logits, dim=-1)


        attention_per_head = torch.matmul(attention_score, V_heads) # [B, num_heads, seq_len, d_head]

        # attention = torch.concat(attention_per_head, dim=1)    # [B, num_heads, seq_len, d_model]
        # attention = attention.permute(0, 2, 1, 3).view(batch_size, self.num_heads, self.d_model)
        '''
        Modified Method:
        '''
        attention = attention_per_head.transpose(1,2).contiguous().view(batch_size, self.d_model)   # [B, seq_len, d_model]


        value = self.o_proj(attention)  ## [B, seq_len, d_model]

        return value

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