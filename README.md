# transformer_from_scratch
I have deeply realized that programming skills are needed even more in the AI era.

So I wanna try to practice and record my progress via this repository.

Considering that I am using an AR model recently and am still kind of confused about the principle behind it, 
I will try to write the code of the transformer from scratch as practice.

## Day 1: Start
1. create the repository and learn some basic operations of 'git'
2. create the virtual environment
3. import some necessary modules, like pytorch.
4. TODO list:
    4.1 Define basic blocks: 
        >> TokenEmbedding
        >> Position Embedding
        >> Multi-Head Attention
        >> FeedForward (Position-wise MLP)
        >> Norm (LayerNorm or RMSNorm)
        >> Residual
        >> Dropout
    4.2 Define class
        >> EncoderLayer
        >> DecoderLayer
        >> Encoder
        >> Decoder
        >> TransformerModel
5. There are too many modules and class need to be implemented. So I decide to implement those mudules as the data-flow.
    Encoder: input embedding ==> position embedding ==> MultiHead Attention ==> Add & Norm 
             ==> Feed Forward ==> Add & Norm
    Decoder: output embedding ==> position embedding ==> Masked MultiHead Attention ==> Add & Norm
             ==> MultiHead Attention ==> Add & Norm ==> Feed Forward ==> Linear ==> Softmax

## Day 2: MultiHeadAttention
I tried to hand-write the 'MultiHeadAttention' class. Not surprisingly, there are many many mistakes. However, that is the
aim of this project.

The mistakes or gains from this practice are as follow:
    1. the usage of 'contiguous()'.
    2. the smart method of create 'qkv_projection' at one time and chunk it as 3 sub-matrix when using.
    3. 'T' can only be used in 2D tensor, while 'transpose()' is bulit for high dimension tensor.
    4. Using 'map()' to divide Q, K, V into heads
