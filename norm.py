class LayerNorm():
    '''
        Considering that we have a batch of items, which item is a vector with 'dimension' of features.
        But the numerical range (value) of items' features may totally different, which is harmful for stable training.
        Then we normalize those items to the same range via learnable parameters 'mean' and 'variance'.  

        'LayerNorm': normalize all the features belong to the same item in the batch
        'BatchNorm': normalize the same feature in all the items of the batch       
    '''