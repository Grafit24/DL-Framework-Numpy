import numpy as np
from base import Module

class Linear(Module):
    """Classic linear layer - y=wx+b."""
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self._bias = bias
        
        # Xavier initialization
        stdv = 1/np.sqrt(dim_in)
        self.W = np.random.uniform(-stdv, stdv, size=(dim_in, dim_out))
        if self._bias:
            self.b = np.random.uniform(-stdv, stdv, size=dim_out)
        
    def forward(self, input):
        self.output = np.dot(input, self.W)
        self.output += self.b if self._bias else 0
        return self.output
    
    def backward(self, input, grad_output):
        self.grad_W = np.dot(input.T, grad_output)
        grad_input = np.dot(grad_output, self.W.T)
        if self._bias:
            self.grad_b = np.mean(grad_output, axis=0)
        return grad_input
    
    def parameters(self):
        return [self.W, self.b] if self._bias else [self.W]
    
    def grad_parameters(self):
        return [self.grad_W, self.grad_b] if self._bias else [self.grad_W]