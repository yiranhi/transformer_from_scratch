import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    '''
        Note: input token sequence = Q = K = V (Q, K, V are just the copy of input token sequence)    
    
        input token sequence: [sequence, model_dimension]
        Q, K, V: [sequence, model_dimension]
        e.g. consider that we have a sentence composed of 6 words, and each word is converted into
        a 512 dimensions vector.

        softmax(torch.dot(Q [6, 512], K.T [512, 6]) / troch.sqrt(512) ) ==> attention score[6, 6]
        torch.dot(attention score [6, 6], V [6, 512]) ==> attention [6, 512] 
    '''
    def __init__(self, d_model:int, attn_dropout:float=0.0, proj_dropout:float=0.0):
        super().__init__()
        self.d_model = d_model

        self.qkv_proj = nn.Linear(d_model, 3*d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        '''
        Error:
            Don't forget dropout
        '''
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x:torch.Tensor):
        batch_size, seq_len, d_model = x.shape    # [B, seq_len, d_model]
        assert d_model == self.d_model

        Q, K, V = self.qkv_proj(x).chunk(3, dim = -1)   # [B, seq_len, d_model]

        attention_logits = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_model)   # [B, seq_len, seq_len]
        attention_weights = F.softmax(attention_logits, dim = -1)    # [B, seq_len, seq_len]

        '''
        Error:
            dropout layer is placed between the softmax and multiply V
        '''
        attention_weights = self.attn_dropout(attention_weights)
        attention = torch.matmul(attention_weights, V)   # [B, seq_len, d_model]

        value = self.o_proj(attention)  # [B, seq_len, d_model]
        '''
        Error:
            After output projection layer, there also should be a dropout layer
        '''
        value = self.proj_dropout(value)

        return value


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

    def __init__(self, num_heads:int, d_model:int, attn_dropout:float=0.0, proj_dropout:float=0.0):
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

        
        self.o_proj = nn.Linear(d_model, d_model, bias = False)

        '''
        Error:
            Don't forget dropout
        '''
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x:torch.Tensor):
        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model

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
        attention_score = F.softmax(attention_logits, dim=-1)

        '''
        Dropout
        '''
        attention_score = self.attn_drop(attention_score)

        attention_per_head = torch.matmul(attention_score, V_heads) # [B, num_heads, seq_len, d_head]

        # attention = torch.concat(attention_per_head, dim=1)    # [B, num_heads, seq_len, d_model]
        # attention = attention.permute(0, 2, 1, 3).view(batch_size, self.num_heads, self.d_model)
        '''
        Modified Method:
        '''
        attention = attention_per_head.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)   # [B, seq_len, d_model]


        value = self.o_proj(attention)  ## [B, seq_len, d_model]

        '''
        Dropout
        '''
        value = self.proj_drop(value)

        return value

class MaskedMultiHeadAttention(nn.Module):
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
        
        Mask (in GPT-like decoder-only model):
            1. Mask is need in both training and inference (as long as you generate using a GPT-like model starting from
               a prompt, you will need the mask during inference). Although they have different shape, they have the same 
               function, i.e. prevent the decoder state from attending to position that corresponding to the tokens 'in the future'.

            2. Why mask is needed during inference?
                because of the decoder only architecture, the output of decoder layer is tranfered into the next decoder layer.

            
            3. The mask in training and inference are different.
               In Training:
                    the whole sequence is put into the model and do only a forward(), so the mask is fixed with a shape
                    of [B, num_heads, seq_len, seq_len], like following:
                    _________________________
                    | 0  -inf ... -inf -inf |
                    | 0   0   ... -inf -inf |
                    |         ...           |
                    | 0   0   ...   0  -inf |
                    | 0   0   ...   0    0  |
                    ------------------------- 
              In Inference:
                    however, the model predicts tokens one by one during inference. So the model will have n (n > t, t is the input seqence length) times forward().
                    In other words, the causal mask "gets longer" as the sequence grows.
                    	At step 1 (first generated token):
                            query attends only to the first key/value → the mask length = 1.
                            [0]
                        At step 2:
                            the new query attends to 2 past tokens → the effective mask length = 2.
                            [0, 0]
                        ...
                        At step t:
                            the query attends to t tokens (all previous context).
                            So the shape of the attention mask row for the new token is [1, kv_len], where kv_len = t grows with the sequence.
                            [0, 0 ... 0]
                        ...
                        At step n:
                            the mask is [1, n]
                            [0, 0 ... 0, ... 0]
    '''
    def __init__(self, num_heads:int, d_model:int, attn_dropout:float=0.0, proj_dropout:float=0.0):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3*d_model, bias = False)
        self.o_proj = nn.Linear(d_model, d_model, bias = False)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def create_causal_mask(seq_len:int) -> torch.Tensor:
        '''
        Create a causal mask for attention.
        Args:
            seq_len: Length of sequence
        Returns:
            Causal Mask with shape of [seq_len, seq_len]
        '''
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1) # [seq_len, seq_len]
        mask = mask.unsqueeze(0).unsqueeze(0)   # [1, 1, seq_len, seq_len]
        return mask
    
    def forward(self, x:torch.Tensor):
        batch_size, seq_len, d_model = x.shape
        assert self.d_model == d_model

        Q, K, V = self.qkv_proj(x).chunk(3, dim=-1) # [B, seq_len, d_model]

        def to_heads(x):
            return x.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1,2)
        Q_heads, K_heads, V_heads = map(to_heads, (Q, K, V))    #[B ,num_heads, seq_len, d_heads]

        attn_weights = torch.matmul(Q_heads, K_heads.transpose(-1, -2)) / math.sqrt(self.d_head)    # [B, num_heads, seq_len, seq_len]

        causal_mask = self.create_causal_mask(seq_len)

        attn_weights = attn_weights + causal_mask

        attn_weights = torch.softmax(attn_weights, dim=-1)

        attn_weights = self.attn_drop(attn_weights)

        attn_per_head = torch.matmul(attn_weights, V_heads) # [B, num_heads, seq_len, d_heads]

        attention = attn_per_head.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model) # [B, seq_len, d_model]

        attention = self.o_proj(attention)

        attention = self.proj_drop(attention)

        return attention