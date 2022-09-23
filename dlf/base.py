import numpy as np
from abc import ABC, abstractmethod


class Module(ABC):
    """Base class - every layer must implement him."""
    def __init__(self):
        self._train = True

    def __call__(self, x):
        return self.forward(x)
    
    @abstractmethod
    def forward(self, input):
        raise NotImplementedError

    @abstractmethod
    def backward(self,input, grad_output):
        raise NotImplementedError
    
    def parameters(self):
        """Returns list of self-parameters."""
        return []
    
    def grad_parameters(self):
        """Returns list of tensor-gradients."""
        return []
    
    def train(self):
        self._train = True
    
    def eval(self):
        self._train = False


class Criterion(ABC):
    """Base class for loss functions."""
    def __call__(self, input, target):
        return self.forward(input, target)
    
    @abstractmethod
    def forward(self, input, target):
        raise NotImplementedError

    @abstractmethod
    def backward(self, input, target):
        raise NotImplementedError


class Optimizer(ABC):
    @abstractmethod
    def __call__(self, params, gradients):
        for weights, gradient in zip(params, gradients):
            ...
            

class Sequential(Module):
    """Group of modules."""
    def __init__ (self, *layers):
        super().__init__()
        self.layers = layers

    def forward(self, input):
        """Forward pass throgh every layer."""

        for layer in self.layers:
            input = layer.forward(input)

        self.output = input
        return self.output

    def backward(self, input, grad_output):
        """Backward pass through every layer."""
        
        for i in range(len(self.layers)-1, 0, -1):
            grad_output = self.layers[i].backward(self.layers[i-1].output, grad_output)
        
        grad_input = self.layers[0].backward(input, grad_output)
        
        return grad_input
      
    def parameters(self):
        """Concat params in one list and return it."""
        res = []
        for l in self.layers:
            res += l.parameters()
        return res
    
    def grad_parameters(self):
        """Concat gradients in one list and return it."""
        res = []
        for l in self.layers:
            res += l.grad_parameters()
        return res
    
    def train(self):
        for layer in self.layers:
            layer.train()
    
    def eval(self):
        for layer in self.layers:
            layer.eval()