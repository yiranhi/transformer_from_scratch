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
![transformer](https://github.com/yiranhi/transformer_from_scratch/blob/main/images/transformer.png)

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

![norm](https://github.com/yiranhi/transformer_from_scratch/blob/main/images/Norm.png "Normalization1")


## Day 6: Feed-Forward 
It's the main and the most obvious place where we add non-linearity(ReLU) to the model, which can give model more flexibility and make it learn non-linear relationship of those inputs.

Feed-Forward operates **position-wise**, meaning it runs independently on each position(token). They won't mix in FFN. Attention mechanism is responsible for gathering all the information from all tokens in the sequence.

The output of attention layer (with shape [batch, seq_len, dimension]) is fed into the FFN. The FFN applies its **identical** weight matrices **independently** to **each** of the 'seq_len' tokens.(This is the meaning of ***Position-Wise***)

## Day 7: Input Embedding &  Position Embedding
**Input Embedding:**

Input Embedding is to transfer the input sentence into the form (vector/tensor) which the follow part of the model can handel. 

The general process is: sentence == tokenizer ==> token id sequence == embedding layer ==> tensor sequence

Note: *The Difference between Embedding Layer & Linear Layer*

&emsp; From the perspective of output, it seems like nn.Embedding and nn.Linear do the same things, i.e. project the input(token ids for nn.Embedding, input tensor for nn.Linear) into a outpu tensor.

&emsp; However, nn.Embedding(num_embeddings, embedding_dim) represents a weight matrix with shape [num_embeddings, embedding_dim]. *num_embeddings* is the vocabulary's size, definig how many tokens can it transfer to tensor. *embedding_dim* represents the dimension of the transfered tensor. From this view, *nn.Embedding* is a actually a lookup table. The input is an index, and it returns the row at the corresponding index in the weight matrix.

&emsp; Remerber the weight matrix of nn.Embedding is also a learnable matrix. If you are building a model from scratch, the weight matrix is initialized randomly (using a normal or uniform distribution). After training (forward pass ==> loss calculation ==> backpropagation), the random weight metrix organize themselves so that similar words end up having similar mathematical vectors. 

&emsp; the class definition are different
"""
class EmbeddingLayer(nn.Module)
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, input):
        return self.weight[input]

class LinearLayer(nn.Module):
    '''
        For CPU/GPU's memory locality optimizations, the weight matrix is usually the shape of [output_dim, input_dim].
        And the forward operation is x @ Weight.T + bias
    '''
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(output_dim, input_dim))
        self.bias = torch.nn.Parameter(torch.randn(output_dim))

    def forward(self, input):
        return torch.matmul(input, self.weight.T) + self.bias
"""


**Position Embedding:**
Since transformer does not process the sequential data (as it lacks recurrent structures), positional information are add to the input token sequence (p.s. literally "add", it's a additive operation).

Let's think about what requirements the position embedding needs to meet.

- The range must be bounded. Otherwise, the position embedding of the last word is so much larger then the first position embedding. After add with input embedding, there will inevitably be a skew in the numerical value of features.

- The position embedding must be unique adn contains the location information, i.e. the order of the contents.

- Encoding difference should not depend on the text length. If using this method, the relative relationship between two words will be diluted in long context. The model would struggle to generalize patterns learned from short sentences to long ones. Such as "cute dog" in a normalized position embedding is [0.33, 0.66]. While the sentence gets longer, like "I have a cute dog", the position embedding will be [0.1, 0.2, 0.3, 0.4, 0.5]. The relative distance of same word "cute" and  "dog" in sentences of different lengths is different.

![norm](https://github.com/yiranhi/transformer_from_scratch/blob/main/images/position_embedding.png)

Based on those requirements above, the transformer uses the periodic function to encode position. Suppose you have an input sequence of Length *L* and require the position of the 
k^th^ object within this sequence.

$$
\begin{aligned}
P(k, 2i) &= \sin\left( \frac{k}{n^{2i/d}} \right) \\
P(k, 2i+1) &= \cos\left( \frac{k}{n^{2i/d}} \right)
\end{aligned}
$$

Here:

- *k*: Position of an object in the input sequence, $0 \lt k \lt \frac{L}{2}$. 

- *d*: Dimension of the output embedding space, $2i \le d$

- *P(k, j)*: Position function for mapping a postion *k* in the input sequence to index (*k, j*) of the position matrixposition. And the author define the odd and even position with cos and sin function separately.

- *n*: User-defined scalar, set to 10,000 by the author of transformer.

![norm](https://github.com/yiranhi/transformer_from_scratch/blob/main/images/pm.png)

This scheme for position embedding has a number of advantages:

1. **Bounded Range** The sine and cosine functions have values in [-1, 1], which keeps the value of the positional encoding matrix in a normalized range.

2. **Unique and Containing Order Information** As the sinusoid for each position(*k*) is different, each position has a unique embedding.

3. **Relative Position Embedding is invariable** You can measure or quantify the similarity between different positions, hence enabling you to encoder the relative positions of words.


## Day 8: Tokenizer
I'd highly recommend that watching Andrej Karpathy's video ((https://www.youtube.com/watch?v=zduSFxRajkE)), which contains everything you need to know about tokenizer.

The tokenizer is a completely separate object from the large language model. Training tokenizer is a completely seaprate pre-rocessing stage with its own training data (probably different from LLM training set).
