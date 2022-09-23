import numpy as np
from dlf.base import Criterion


class MSE(Criterion):
    def forward(self, input, target):
        batch_size = input.shape[0]
        self.output = np.sum(np.power(input - target, 2)) / batch_size
        return self.output
 
    def backward(self, input, target):
        self.grad_output  = (input - target) * 2 / input.shape[0]
        return self.grad_output


class CrossEntropy(Criterion):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, target): 
        # чтобы нигде не было взятий логарифма от нуля:
        eps = 1e-9
        input_clamp = np.clip(input, eps, 1 - eps)
        self.output = (-np.log(input_clamp) * target).sum(axis=1)
        return self.output

    def backward(self, input, target):
        eps = 1e-9
        input_clamp = np.clip(input, eps, 1 - eps)
        grad_input = -target * 1/input_clamp
        return grad_input