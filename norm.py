import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    '''
        Considering that we have a batch of items, which item is a vector with 'dimension' of features.
        But the numerical range (value) of items' features may totally different, which is harmful for stable training.
        Then we normalize those items to the same range via 'mean' and 'variance'.  

        'LayerNorm': normalize all the features belong to the same item in the batch
        'BatchNorm': normalize the same feature in all the items of the batch       
    '''
    def __init__(self, dim, eps:float=1e-6):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # x.shape [batch, seq_len, dimension]
        mean = x.mean(dim=-1, keepdim =True)    # [batch, seq_len, 1]
        var = x.var(dim=-1, keepdim=True, unbiased=False)   # [batch, seq_len, 1]
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weights * x_norm + self.bias

class RMSNorm(nn.Module):
    def __init__(self, dim, eps:float=1e-6):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(dim))    # torch.size([dim, ])

    def forward(self, x):
        # batch, seq_len, dim = x.shape
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)   # torch.size([batch, seq_len, 1])
        x_norm = self.weights * x * rms

        return x_norm

class BatchNorm(nn.Module):
    '''
    Why need momentum?
        we use momentum to weighted sum the mean (batch_mean) and var(batch_var) from each batch, 
        so that we can get the mean(running_mean) and var(running_var) representing the whole training set. 
        Then 'running_mean' and 'running_var' can be used in inference.

        {
            在训练时，使用当前batch的mean和var来对这批数据进行归一化。这个过程是动态的，每一批数据的mean和var都在变化。
            （我们对每个batch的均值和方差，进行加权更新，最终得到整个数据集的平均mean和var。）
            
            在推理的时候，我们可能只想推理一个样本。batch size为1，那么计算单一样本的mean和var毫无意义。因此，我们不能
            依靠当前batch的mean和var，而要用一个固定的、代表整个训练集分布的均值和方差。
        }

    batch_mean, batch_var:
        is the mean and var calculated from the current batch
        batch_mean = mean(current_batch), batch_var = var(current_batch)

    running_mean, running_var:
        is the mean and var globally accumulated estimated over the entire training process.
        running_mean = (1 - momentum) * running_mean + momentum * batch_mean
        running_var = (1 - momentum) * running_var + momentum * batch_var
    '''
    def __init__(self, dim, momentum:float=0.1, eps:float=1e-6):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(dim))    # [C, ]
        self.bias = nn.Parameter(torch.zeros(dim))      # [C, ]

        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var', torch.ones(dim))

        self.momentum = momentum

    def forward(self, x):
        # x.shape [B, C, H, W]
        if self.training:
            batch_mean = x.mean(dim=[0, 2, 3])  # [C, ]
            batch_var = x.var(dim=[0, 2, 3])    # [C, ]

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var

        # Important!
        # the mean and var have to align with x's normalized dimension, so taht they can be broadcasted to x
        batch_mean = batch_mean.view(1, -1, 1, 1)   # [1, C, 1, 1]
        batch_var = batch_var.view(1, -1, 1, 1) # [1, C, 1, 1]
        self.weghts = self.weights.view(1, -1, 1, 1)
        self.bias = self.bias.view(1, -1, 1, 1)
        
        x_norm = (x - batch_mean) * (batch_var + self.eps).rsqrt() * self.weights + self.bias

        return x_norm 
        
