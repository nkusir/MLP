"""

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

>>> activation = Identity()
>>> activation(3)
3
>>> activation.forward(3)
3
"""

import numpy as np
import os



class Activation(object):
    """ Interface for activation functions (non-linearities).

        In all implementations, the state attribute must contain the result, i.e. the output of forward (it will be tested).
    """

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):
    """ Identity function .
     """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):
    """ Implement the sigmoid non-linearity """

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        self.state = 1 / (1 + np.exp(-x ))
        
        return self.state

    def derivative(self):
        self.deriv = self.state*(1 - self.state)
        return self.deriv


class Tanh(Activation):
    """ Implement the tanh non-linearity """

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state = np.tanh(x)
        return self.state

    def derivative(self):
        self.deri = 1 - self.state**2
        
        return self.deri


class ReLU(Activation):
    """ Implement the ReLU non-linearity """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.x = x
        x [x < 0] = 0
        self.state = x
        return self.state

    def derivative(self):
         y = self.x
         y[y < 0] = 0 
         y[y > 0] = 1
         return y


# CRITERION


class Criterion(object):
    """ Interface for loss functions.
    """

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

# Implementing the loss function
class SoftmaxCrossEntropy(Criterion):
    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None
# Forward pass
    def forward(self, x, y):
        self.logits = x
        exps = np.exp(x - np.max(x))
        x = exps / np.sum(exps,axis=1).reshape(exps.shape[0],1)
        self.sm = x
        self.y = y
        
        s = np.log(x) 
        self.loss = - np.sum((y*s),axis = 1)
        
        
        return self.loss
# Derivative of The loss function wrt output
    def derivative(self):
        self.deriva = np.subtract(self.sm,self.y)
        #print(self.deriva.shape)
        return self.deriva

# Implementing Batch normalization , refer to the Batch normalization paper for formulas and notation
class BatchNorm(object):
    def __init__(self, fan_in, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        # inference parameters
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)
	
	# Forward pass with Batch Normalization
    def forward(self, x, eval=False):
        self.x = x
        self.n_X = x.shape[0]
        self.X_shape = x.shape
        self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * self.mean
        self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.var
        self.x = x.ravel().reshape(self.n_X,-1)
        self.mean = np.mean(x, axis=0)
        self.var = np.var(x, axis=0)
        self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        out = self.gamma * self.x_norm + self.beta
        if eval == True:
            out = (self.gamma/np.sqrt(self.running_var + self.eps))*self.x_norm + (self.beta - (self.gamma*self.running_mean /np.sqrt(self.running_var + self.eps))) 
        return out
		
	# Backward with Batch normalization
    def backward(self, delta):
        delta = delta.ravel().reshape(delta.shape[0],-1)

        N, D = self.x.shape

        X_mu = self.x - self.mean
        std_inv = 1. / np.sqrt(self.var + 1e-8)

        dX_norm = delta * self.gamma
        dvar = np.sum(dX_norm * X_mu, axis=0) * -.5 * std_inv**3
        dmu = np.sum(dX_norm * -std_inv, axis=0) + dvar * np.mean(-2. * X_mu, axis=0)

        dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)
        self.dgamma = np.sum(delta * self.x_norm, axis=0)
        self.dbeta = np.sum(delta, axis=0)
        return dX


def random_normal_weight_init(d0, d1):
    raise NotImplemented

def zeros_bias_init(d):
    raise NotImplemented


class MLP(object):
    """ A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens,
                 activations, weight_init_fn, bias_init_fn,
                 criterion, lr, momentum, num_bn_layers=0):
        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        self.weight_init_fn = weight_init_fn
        self.bias_init_fn =bias_init_fn
        self.hiddens = hiddens
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes
        self.W =[]
        self.b =[]
        self.dW =[]
        self.db =[]
        self.dw = []
        self.dB = []
        self.bb = []
        self.ww = []
        self.bn_layers = []
        if self.bn:
            self.bn_layers = []
        
		# Initializing the weights and bias in case we have more than one layer
        if self.nlayers >1 :
            self.W.append(self.weight_init_fn(self.input_size,self.hiddens[0]))
            self.dW.append(np.zeros_like(self.W[0]))
            self.dw.append(np.zeros_like(self.W[0]))
            self.ww.append(np.zeros_like(self.W[0]))
            self.b.append(self.bias_init_fn(self.hiddens[0]))
            self.d = BatchNorm(self.W[0].shape[1])
            self.bn_layers.append(self.d)
            self.db.append(np.zeros_like(self.b[0]))
            self.dB.append(np.zeros_like(self.b[0]))
            self.bb.append(np.zeros_like(self.b[0]))
            for i in range( 1,self.nlayers -1):
                self.W.append(self.weight_init_fn(self.hiddens[i-1],self.hiddens[i ]))
                self.dW.append(np.zeros_like(self.W[i]))
                self.dw.append(np.zeros_like(self.W[i]))
                self.ww.append(np.zeros_like(self.W[i]))
                self.b.append( self.bias_init_fn(self.hiddens[i ]))
                self.d = BatchNorm(self.W[i].shape[1])
                self.bn_layers.append(self.d)
                self.db.append(np.zeros_like(self.b[i]))
                self.dB.append(np.zeros_like(self.b[i]))
                self.bb.append(np.zeros_like(self.b[i]))
            self.W.append(self.weight_init_fn(self.hiddens[self.nlayers-2],self.output_size))
            self.b.append( self.bias_init_fn(self.output_size))
            self.db.append( np.zeros_like(self.b[self.nlayers-1]))
            self.d = BatchNorm(self.W[self.nlayers -1].shape[1])
            self.bn_layers.append(self.d)
            self.dW.append(np.zeros_like(self.W[self.nlayers-1]))
            self.dB.append( np.zeros_like(self.b[self.nlayers-1]))
            self.dw.append(np.zeros_like(self.W[self.nlayers-1]))
            self.bb.append( np.zeros_like(self.b[self.nlayers-1]))
            self.ww.append(np.zeros_like(self.W[self.nlayers-1]))
        else:
            self.W.append(self.weight_init_fn(self.input_size,self.output_size))
            self.dW.append(np.zeros_like(self.W[0]))
            self.dw.append(np.zeros_like(self.W[0]))
            self.ww.append(np.zeros_like(self.W[0]))
            self.d = BatchNorm(self.W[0].shape[1])
            self.bn_layers.append(self.d)
            
            self.b.append(self.bias_init_fn(self.output_size))
            self.db.append(np.zeros_like(self.b[0]))
            self.dB.append(np.zeros_like(self.b[0]))
            self.bb.append(np.zeros_like(self.b[0]))

	# Performing THe forward pass on any number of layer taking into consideration the batch Normalization
	# collecting all output at each layer
    def forward(self, x):
        self.x1 = x
        self.y1 = [x]
        self.indy1 = 0
        
        for i in range(self.nlayers):
            self.forwardNohiddenLayers = np.matmul(x,self.W[i]) + self.b[i]
            
            if  self.num_bn_layers != 0:
                if i < 1 :
                    
                
                    for j in range(1):
                        
                        self.forwardNohiddenLayers = self.bn_layers[j].forward(self.forwardNohiddenLayers)
            
            self.outForwardNHL = self.activations[i](self.forwardNohiddenLayers)
            
            self.y1.append(self.outForwardNHL)
            x = self.outForwardNHL
        return self.outForwardNHL
	
	# initializing gradients wrt weigth and bias to zero

    def zero_grads(self):
      
        
        for i in range(self.nlayers):
            self.dW.append( np.zeros_like(self.W[i]))
            self.db.append(np.zeros_like(self.b[i]))
            
        
    def step(self):
        
        for i in range(self.nlayers): 
            # updating weights and bias after backprop
            self.W[i] =self.W[i] - self.lr*self.dW[i]
            self.b [i]= self.b[i] - self.lr*self.db[i]
            
			# updating batch norm parameters afer backprop
            if  self.num_bn_layers !=0:
                if i < 1:
                    
                    self.bn_layers[i].gamma = self.bn_layers[i].gamma - self.lr*self.bn_layers[i].dgamma
                    self.bn_layers[i].beta = self.bn_layers[i].beta - self.lr*self.bn_layers[i].dbeta

                
	#Perfoming backprop for any number of layers
    def backward(self, labels):
        self.labels = labels
        
        if self.nlayers == 1:
            lossD = self.criterion(self.outForwardNHL , self.labels)
            der = self.criterion.derivative()
            r = self.activations[0].derivative()*der
            if self.num_bn_layers != 0:
                for j in range(1):
                    
                    r = self.bn_layers[0].backward(r)
            self.dW[0] = (np.dot(self.x1.T,r) /len(labels))
            
            self.db[0] = ( np.sum(r,0) / len(labels))
            self.dB[0] =((self.momentum*self.dB[0] -  (1- self.momentum)*self.db[0]))
            self.dw[0] = ((self.momentum*self.dw[0] -  (1- self.momentum)*self.dW[0]))
        else:
            self.dW = []
            self.db = []
            
            lossD = self.criterion(self.outForwardNHL , self.labels)
            der = self.criterion.derivative()

            r = self.activations[self.nlayers -1].derivative()*der

            for g in range(self.nlayers-1):
                r = self.activations[self.nlayers -1].derivative()*der
                for i in range(self.nlayers-1,g,-1):
                    
                     

                    s = r.dot(self.W[i].T)
	                    
                    r = s*self.activations[i -1].derivative()
                
                
                
                
                if  self.num_bn_layers !=0:
                    if g < 1:
                        
                        for j in range(1):
                            r = self.bn_layers[j].backward(r)
                
                s = self.y1[g].T.dot(r)
                
                self.dW.append( s/ len(labels))
                
                
                self.dw[g]= ((self.momentum*self.dw[g] -  (1- self.momentum)*self.dW[g]) )
                
                self.db.append( np.sum(r,0)/len(labels))
                self.dB[g] = ((self.momentum*self.dB[g] -  (1- self.momentum)*self.db[g]))
                self.indy1+=1
           
        
        return lossD
        

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False





