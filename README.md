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
&emsp;Encoder: input embedding ==> position embedding ==> MultiHead Attention ==> Add & Norm 
             ==> Feed Forward ==> Add & Norm<br> 
&emsp;Decoder: output embedding ==> position embedding ==> Masked MultiHead Attention ==> Add & Norm
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
**'MaskedMUltiHeadAttention'** one. The mask used in **'training'** is easy to understand. Because the input sequence contains all
the information, and you have to prevent the decoder state from attending to the position that corresponding to the tokens
'in the future'. In other word, 'peek ahead'.

However,It's kind of confused for me to understand the mask used in **'inference'**. I believe the mask should be used during
inference, although I haven't find the perfect way to explain it.
(This video 'Masking During Transformer Inference Matters a Lot (Buy Why?) - YouTube' doesn't really let me understand the 
principle behind the inference mask.)

Somebody says that **KV cache** is the key point of understranding the mask in infernece.
If you are interested in KV cache, please refer to (https://huggingface.co/blog/not-lain/kv-caching)

## Day 4: KV cache
