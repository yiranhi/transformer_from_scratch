# transformer_from_scratch
I have deeply realized that programming skills are needed even more in the AI era.

So I wanna try to practice and record my progress via this repository.

Considering that I am using an AR model recently and am still kind of confused about the principle behind it, 
I will try to write the code of the transformer from scratch as practice.
## Day 1: Start
1. create the repository and learn some basic operations of 'git'
2. create the virtual environment
3. import some necessary modules, like pytorch.
4. TODO list:<br>
&emsp;4.1 Define basic blocks:<br> 
&emsp;&emsp;>> TokenEmbedding<br> 
&emsp;&emsp;>> Position Embedding<br> 
&emsp;&emsp;>> Multi-Head Attention<br> 
&emsp;&emsp;>> FeedForward (Position-wise MLP)<br> 
&emsp;&emsp;>> Norm (LayerNorm or RMSNorm)<br> 
&emsp;&emsp;>> Residual<br> 
&emsp;&emsp;>> Dropout<br> 
&emsp;4.2 Define class<br> 
&emsp;&emsp;>> EncoderLayer<br> 
&emsp;&emsp;>> DecoderLayer<br> 
&emsp;&emsp;>> Encoder<br> 
&emsp;&emsp;>> Decoder<br> 
&emsp;&emsp;>> TransformerModel<br> 
5. There are too many modules and class need to be implemented. So I decide to implement those mudules as the data-flow.<br> 
&emsp;**Encoder:** 
&emsp;&emsp;input embedding ==> position embedding ==> MultiHead Attention ==> Add & Norm 
             ==> Feed Forward ==> Add & Norm<br> 
&emsp;**Decoder:** 
&emsp;&emsp;output embedding ==> position embedding ==> Masked MultiHead Attention ==> Add & Norm
             ==> MultiHead Attention ==> Add & Norm ==> Feed Forward ==> Linear ==> Softmax<br> 
## Day 2: MultiHeadAttention
I tried to hand-write the **'MultiHeadAttention'** class. Not surprisingly, there are many many mistakes. However, that is the
aim of this project.

The mistakes or gains from this practice are as follow:<br>
&emsp;1. the usage of **'contiguous()'**.<br>
&emsp;2. the smart method of create **'qkv_projection'** at one time and chunk it as 3 sub-matrix when using.<br>
&emsp;3. **'T'** can only be used in 2D tensor, while **'transpose()'** is bulit for high dimension tensor.<br>
&emsp;4. Using **'map()'** to divide Q, K, V into heads<br>
## Day 3: Mask in train and inference
Considering I have finished building **'SelfAttention'** class and **'MultiHeadAttention'** class yesterday, I try to finish the 
**'MaskedMUltiHeadAttention'** one. 

The mask used in **'training'** is easy to understand. Because the input sequence contains all
the information, and you have to prevent the decoder state from attending to the position that corresponding to the tokens
'in the future'. In other word, 'peek ahead'.

However,It's kind of confused for me to understand the mask used in **'inference'**. I believe the mask should be used during
inference, although I haven't find the perfect way to explain it.

This video 'Masking During Transformer Inference Matters a Lot (Buy Why?)'(https://www.youtube.com/watch?v=ZARUDeRhEwc) doesn't really let me understand the principle behind the inference mask.

Somebody says that **KV cache** is the key point of understranding the mask in infernece (But I am still confused). KV cache is a technique which is used during inference.
It solves recomputation by storing the attnetion from previous steps.

If you are interested in KV cache, please refer to (https://huggingface.co/blog/not-lain/kv-caching)

*update:* I tried to understand the **'mask in inference'** from another prospective. In Decoder-only architecture like GPT, they remove the **'multi-Head Attention'** because they don't have encoder's output. And in GPT-like models, they have a sequence as input. However, the docoder have to do *n* times forward and predict tokens one by one, which means at least before the last token in input sequence was generated, the decoder still does **teacher forcing** as training. So the mask is needed in inference.

*Note:* This is my opinion about mask in inference. I am not sure it's correct or not. Please leave your comments and let's have deeper understanding about this question.

## Day 4: The Implement of 'MaskedMultiHeadAttention'
The class **SelfAttention**, **MultiHeadAttention** and **MaskedMultiHeadAttention** are in the *multi_head_attention.py*.

To create a causal mask, you will need 2 helper functions - **'torch.full()'** and **'torch.triu()'**

    torch.triu(input, diagonal=0, *, out=None)
        function: returns the upper triangular part of a matrix (2D tensor) or batch of matrics *input*, the other elements are set as *out*, 0 as default.

        args:
            diagonal:
                diagonal=0, all elements on and above the main diagonal are retained. 
                diagonal=1, the elements above the main diagonal are retained.
                diagonal=-1, the elements below the main diafonal are retained.


## Day 5: Add & Norm
***Add*** means residual connection, which solves the vanishing gradient problem allowing deeper model. (More detail about Residual, see (https://arxiv.org/abs/1512.03385)) Residual connection has almost become a default setting for models.

***Normalization*** is another technique used to stabilize and accelerate training process. Transformer uses **Layer Norm**.
However, there are 3 main normalization methods mostly used: Layer Norm, Batch Norm, RMS Norm.<br>

&emsp;***Layer Norm***: normalizes all features within each sample, which eliminates the magnitude difference between different samples but preserves the relative magnitude relationships between different features within a single sample. (Commonly used in the field of NLP)<br>
&emsp;&emsp;*Example:*

    # NLP
    batch, sentence_length, embedding_dim = input.shape
    layer_norm = nn.LayerNorm(embedding_dim)
    layer_norm(input)

    # Image
    B, C, H, W = input.shape
    layer_norm = nn.LayerNorm([C, H, W])
    layer_norm(input)


&emsp;***Batch Norm***: normalizes each features across all samples in a batch, which eliminates the magnitude differences between different features but preserves the relative magnitude relationships between different samples. (Commonly used in the field of CV)<br>
(*Note*: **momentum** is a hyperparameter that creates the mean and var for the whole training set. I will explain it in the code [norm](./norm.py))<br>
&emsp;&emsp;*Example:*

    # Image
    B, C, H, W = input.shape
    batch_norm = nn.BatchNorm2d(C)
    batch_norm(input)

&emsp;***RMS Norm***: is a simplization of Layer Norm. It's taken over the last dimensions.

Note: **register_buffer('name', tensor)**<br>
&emsp;If you have parameters in your model, which should be saved and restored in 'state_dict', but not trained by the optimizer, you should regiseter them as buffer.

&emsp;As the 'momentum' in batch norm, which updates with the mean and var from mini-batch, doesn't depend on optimizer.

Note:<br>
&emsp;Remeber to care about the dimension of mean, var, weights, bias. Make sure they can be broadcasted to the same dimension of input.

![norm](./images/norm.png "Normalization1")

## Day 5: Feed-Forward 


## Day 6: Position Embedding & Input Embedding
