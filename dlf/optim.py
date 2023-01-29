import numpy as np
from dlf.base import Optimizer


class SGD(Optimizer):
    def __init__(self, lr=1e-3):
        self.lr = lr

    def __call__(self, params, gradients):
        for weights, gradient in zip(params, gradients):
            weights -= self.lr * gradient


class Momentum(Optimizer):
    def __init__(self, lr=1e-3, momentum=.9):
        self.lr = lr
        self.momentum = momentum
        self._u = []

    def __call__(self, params, gradients):
        if len(self._u) == 0:
            self._u = [0]*len(params)
        for i, (weights, gradient) in enumerate(zip(params, gradients)):
            self._u[i] = self._u[i]*self.momentum + self.lr*gradient
            weights -= self._u[i]


class RMSprop(Optimizer):
    def __init__(self, lr=1e-3, eps=1e-8, beta=.9):
        self.lr = lr
        self.eps = eps
        self.beta = beta
        self._g = []

    def __call__(self, params, gradients):
        if len(self._g) == 0:
            self._g = [0]*len(params)
        for i, (weights, gradient) in enumerate(zip(params, gradients)):
            self._g[i] = self._g[i]*self.beta + gradient*gradient*(1-self.beta)
            weights -= (self.lr*gradient)/np.sqrt(self._g[i] + self.eps)


class Adam(Optimizer):
    def __init__(self, lr=1e-3, beta1=.9, beta2=.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m = []
        self._u = []
        self.t = 1

    def __call__(self, params, gradients):
        if (len(self._u) == 0) and (len(self._m) == 0):
            self._u = [0]*len(params)
            self._m = [0]*len(params)
        for i, (weights, gradient) in enumerate(zip(params, gradients)):
            self._m[i] = self.beta1*self._m[i] + (1-self.beta1)*gradient
            self._u[i] = self.beta2*self._u[i] + (1-self.beta2)*gradient*gradient
            hat_m = self._m[i]/(1-self.beta1**self.t)
            hat_u = self._u[i]/(1-self.beta2**self.t)
            weights -= (self.lr*hat_m)/(np.sqrt(hat_u) + self.eps)
        self.t += 1

    def reset_t(self):
        self.t = 1


class NAdam(Optimizer):
    def __init__(self, lr=1e-3, beta1=.9, beta2=.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m = []
        self._u = []
        self.t = 1

    def __call__(self, params, gradients):
        if (len(self._u) == 0) and (len(self._m) == 0):
            self._u = [0]*len(params)
            self._m = [0]*len(params)
        for i, (weights, gradient) in enumerate(zip(params, gradients)):
            self._m[i] = self.beta1*self._m[i] + (1-self.beta1)*gradient
            self._u[i] = self.beta2*self._u[i] + (1-self.beta2)*gradient*gradient
            hat_m = self._m[i]/(1-self.beta1**self.t)
            hat_u = self._u[i]/(1-self.beta2**self.t)
            weights -= (self.lr \
                *(self.beta1*hat_m + ((1-self.beta1)*gradient)/(1-self.beta1**self.t))) \
                /(np.sqrt(hat_u) + self.eps)
        self.t += 1

    def reset_t(self):
        self.t = 1