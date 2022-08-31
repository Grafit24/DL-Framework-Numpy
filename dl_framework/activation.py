import numpy as np
from base import Module


class ReLU(Module):
    def __init__(self):
         super().__init__()
    
    def forward(self, input):
        self.output = np.maximum(input, 0)
        return self.output
    
    def backward(self, input, grad_output):
        grad_input = np.multiply(grad_output, input > 0)
        return grad_input


class LeakyReLU(Module):
    def __init__(self, slope=0.03):
        super().__init__()
        self.slope = slope
        
    def forward(self, input):
        self.output = (input > 0)*input + (input <= 0)*self.slope*input
        return self.output
    
    def backward(self, input, grad_output):
        grad_input = np.multiply(grad_output, (input > 0) + (input <= 0)*self.slope)
        return grad_input


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.output = self.__class__._sigmoid(input)
        return self.output
    
    def backward(self, input, grad_output):
        sigma = self.__class__._sigmoid(input)
        grad_input = np.multiply(grad_output, sigma*(1 - sigma))
        return grad_input

    @staticmethod
    def _sigmoid(x):
        return 1/(1 + np.exp(-x))


class SoftMax(Module):
    def __init__(self):
         super().__init__()
    
    def forward(self, input):
        self.output = self._softmax(input)
        return self.output
    
    def backward(self, input, grad_output):
        p = self._softmax(input)
        grad_input = p * (grad_output - (grad_output * p).sum(axis=1)[:, None])
        return grad_input

    def _softmax(self, x):
        x = np.subtract(x, x.max(axis=1, keepdims=True))
        e_m = np.exp(x)
        sum_e = np.repeat(np.sum(e_m, axis=1), x.shape[-1]).reshape(*e_m.shape)
        return e_m/sum_e