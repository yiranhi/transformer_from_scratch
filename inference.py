'''
    The model complete inference in N steps (until the <EOS> appears).
    Beacause the model predict the token one by one. When you get the new token,
    you have to create the new sequence (previous sequence + new token) and send it
    to the model.
'''