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
    """Тоже самое, что и Sofrmax + NLL только быстрее. 
    Соответсвенно принимает не распределение вероятностей, a логиты (выход Linear).
    """
    def forward(self, input, target): 
        batch_size = input.shape[0]
        prob = self._softmax(input)
        # чтобы нигде не было взятий логарифма от нуля:
        eps = 1e-9
        prob_clamp = np.clip(prob, eps, 1 - eps)
        self.output = np.sum(-np.log(prob_clamp) * target) / batch_size
        return self.output

    def backward(self, input, target):
        batch_size = input.shape[0]
        prob = self._softmax(input)
        eps = 1e-9
        prob_clamp = np.clip(prob, eps, 1 - eps)
        self.grad_output = (prob_clamp - target) / batch_size
        return self.grad_output