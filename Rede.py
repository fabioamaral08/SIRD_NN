import numpy as np
class Rede(object):
    def __init__(self):
        self.layers = []
    ##----------------------------------------------------------
    def add_lay(self, num_in = None, num_out = None, bias = False, activation = 'Sigmoid'):
        if num_out is None:
            raise NameError('Missing Argument: num_out')
        if num_in is None:
            if len(self.layers) == 0:
                raise NameError('Missing Argument: num_in is required for the first Layer')
            else:
                num_in = self.layers[-1].num_out
        if not callable(activation):
            dct = {
                "Sigmoid": Sigmoid,
                "Linear": Linear,
                "Relu": Relu,
                "Relu_min": Relu_min
            }

            activation = dct[activation]
        l = Layer(num_in = num_in, num_out = num_out, bias = bias, activation = activation)
        self.layers.append(l)
    ##----------------------------------------------------------

    ##----------------------------------------------------------
    
    def run(self, x):
        n_in = self.layers[0].num_in
        H = np.array([x]).reshape((-1,n_in))
        for l in self.layers:
            H = l.run(H)
        return H
    ##----------------------------------------------------------

    ##----------------------------------------------------------

    def get_weights(self):
        w = np.array([])

        for l in self.layers:
            w = np.append(w, l.get_weights())
        return w
    ##----------------------------------------------------------
    ##----------------------------------------------------------
    def set_weights(self, weights):
        i = 0
        for l in self.layers:
            n = l.get_num_param()
            w = weights[i:i+n]
            l.set_weights(w)
            i +=n
    ##----------------------------------------------------------
    ##----------------------------------------------------------
    def get_num_param(self):
        n = 0
        for l in self.layers:
            n += l.get_num_param()
        return n
    ##----------------------------------------------------------
    ##----------------------------------------------------------
    def print_net(self):
        if len(self.layers) == 0:
            print("Empty Module")
            return
        for i,l in enumerate(self.layers):
            print(f'[{i}]')
            l.print_lay()
    ##----------------------------------------------------------


class Layer(object):
    def __init__(self, num_in, num_out, activation, bias = False):
        self.num_in = num_in
        self.num_out = num_out
        self.activation = activation
        self.has_bias = bias

        self.weights = np.zeros((num_in,num_out))
        if self.has_bias:
            self.bias = np.ones((num_out,))
        else:
            self.bias = None
        
    def run(self, x):
        if self.has_bias:
            H =  self.activation(np.dot(x, self.weights) + self.bias)
        else:
            H =  self.activation(np.dot(x, self.weights))
        return (H)


    def get_weights(self):
        w = self.weights.flatten()
        if self.has_bias:
            w = np.append(w,self.bias.flatten())
        return w
    
    def set_weights(self, weights):
        if self.has_bias:
            self.weights = weights[:-self.num_out].reshape((self.num_in, self.num_out))
            self.bias    = weights[-self.num_out:].reshape((self.num_out,))
        else:
            self.weights = weights.reshape((self.num_in, self.num_out))

    def get_num_param(self):
        n = self.num_in * self.num_out
        if self.has_bias:
            n += self.num_out
        return n

    def print_lay(self):
        print(f'Input: {self.num_in}, Output: {self.num_out}, bias: {self.has_bias}, Activation: ', {self.activation})



"""
Activation functions
"""

def Sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def Relu(x):
    return np.maximum(x, 0)

def Relu_min(x):
    return np.maximum(x, 0) + 0.001

def Linear(x):
    return x