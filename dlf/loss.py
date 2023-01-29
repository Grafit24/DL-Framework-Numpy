import numpy as np
from dlf.base import Criterion
from dlf.activation import Softmax


class MSE(Criterion):
    """Mean Squared Error"""
    def forward(self, input, target):
        batch_size = input.shape[0]
        self.output = np.sum(np.power(input - target, 2)) / batch_size
        return self.output
 
    def backward(self, input, target):
        batch_size = input.shape[0]
        self.grad_output  = (input - target) * 2 / batch_size
        return self.grad_output


class NLL(Criterion):
    """Negative log-likelihood"""
    def forward(self, input, target):
        batch_size = input.shape[0]
        self.output = np.sum(-input * target) / batch_size
        return self.output
    
    def backward(self, input, target):
        batch_size = input.shape[0]
        self.grad_output = -target / batch_size
        return self.grad_output

class CrossEntropy(Criterion, Softmax):
    """The same as Sofrmax + NLL but faster. 
    Accordingly it doesn't accept probability distributions, but logits (output of Linear).
    """
    def forward(self, input, target): 
        batch_size = input.shape[0]
        self.prob = self._softmax(input)
        # for no taking log from zero
        eps = 1e-9
        prob_clamp = np.clip(self.prob, eps, 1 - eps)
        self.output = np.sum(-np.log(prob_clamp) * target) / batch_size
        return self.output

    def backward(self, input, target):
        batch_size = input.shape[0]
        eps = 1e-9
        prob_clamp = np.clip(self.prob, eps, 1 - eps)
        self.grad_output = (prob_clamp - target) / batch_size
        return self.grad_output