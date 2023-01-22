import numpy as np
from dlf.base import Module


class ReLU(Module):
    def forward(self, input):
        self.output = np.maximum(input, 0)
        return self.output
    
    def backward(self, input, grad_output):
        grad_input = np.multiply(grad_output, input > 0)
        return grad_input


class LeakyReLU(Module):
    def __init__(self, slope=0.01):
        super().__init__()
        self.slope = slope
        
    def forward(self, input):
        self.output = (input > 0)*input + (input <= 0)*self.slope*input
        return self.output
    
    def backward(self, input, grad_output):
        grad_input = np.multiply(grad_output, (input > 0) + (input <= 0)*self.slope)
        return grad_input


class Sigmoid(Module):
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


class Tanh(Module):
    def forward(self, input):
        self.output = np.tanh(input)
        return self.output
    
    def backward(self, input, grad_output):
        th = np.tanh(input)
        grad_input = np.multiply(grad_output, (1 - th*th))
        return grad_input


class Softmax(Module):
    def forward(self, input):
        self.output = self._softmax(input)
        return self.output
    
    def backward(self, input, grad_output):
        p = self._softmax(input)
        grad_input = p * ( grad_output - (grad_output * p).sum(axis=1)[:, None] )
        return grad_input

    def _softmax(self, x):
        x = np.subtract(x, x.max(axis=1, keepdims=True))
        e_m = np.exp(x)
        sum_e = np.repeat(np.sum(e_m, axis=1), x.shape[-1]).reshape(*e_m.shape)
        return e_m/sum_e


class LogSoftmax(Softmax):
    def forward(self, input):
        # чтобы нигде не было взятий логарифма от нуля:
        eps = 1e-9
        softmax_clamp = np.clip(self._softmax(input), eps, 1 - eps)
        self.output = np.log(softmax_clamp)
        return self.output

    def backward(self, input, grad_output):
        return (1/self._softmax(input)) * super().backward(input, grad_output)