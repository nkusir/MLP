# MLP
The parameters for the MLP class are: 
• input size: The size of each individual data example. 
• output size: The number of output logits. 
• hiddens: A list with the number of units in each hidden layer. 
• activations: A list of Activation objects for each layer. 
• weight init fn: A function applied to each weight matrix before training. 
• bias init fn: A function applied to each bias vector before training.
• criterion: A Criterion object to compute the loss and its derivative. 
• lr: The learning rate. 
• momentum: Momentum scale (Should be 0.0 until completing 2.6). 
• num bn layers: Number of BatchNorm layers start from upstream (Should be 0 until completing 2.5). 

The attributes of the MLP class are: 
• @W: The list of weight matrices. 
• @dW: The list of weight matrix gradients. 
• @b: A list of bias vectors. 
• @db: A list of bias vector gradients. 
• @bn layers: A list of BatchNorm objects. (Should be None until completing 2.5).


Implement the softmax cross entropy operation on a batch of logit vectors.
Hint: Add a class attribute to keep track of intermediate values necessary for the backward computation. 
• Input shapes: 
    – x: (batch size, 10)
    – y: (batch size, 10) 
• Output Shape: 
    – out: (batch size,)


Perform the ‘backward’ pass of softmax cross entropy operation using intermediate values saved in the 
forward pass. 
• Output shapes: 
    – out: (batch size, 10)
