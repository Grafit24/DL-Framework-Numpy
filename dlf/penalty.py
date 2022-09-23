import numpy as np
from dlf.base import Module


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        
        self.p = p
        self.mask = None
        
    def forward(self, input):
        if self._train:
            p_save = 1 - self.p
            self.mask = np.random.binomial(1, p=p_save, size=input.shape)/p_save
            self.output = self.mask*input
        else:
            self.output = input
        return self.output
    
    def backward(self, input, grad_output):
        if self._train:
            grad_input = self.mask*grad_output
        else:
            grad_input = grad_output
        return grad_input


class BatchNorm(Module):
    def __init__(self, num_features, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        self.sigma_mean = 1
        self.mu_mean = 0
    
    def forward(self, input):
        if self._train:
            assert input.shape[0] > 1, "Batch size need to be >1"
            self._mu = np.mean(input, axis=0)
            self._sigma = np.var(input, axis=0)
            self.mu_mean = self.mu_mean*.9 + self._mu*.1 
            self.sigma_mean = self.sigma_mean*.9 + self._sigma*.1
            self._input_norm = self._normalize(input, self._mu, self._sigma)
            self.output = self.gamma*self._input_norm + self.beta
        else:
            self._input_norm = self._normalize(input, self.mu_mean, self.sigma_mean)
            self.output = self.gamma*self._input_norm + self.beta
        return self.output
    
    def backward(self, input, grad_output):
        if self._train:
            m = input.shape[0]
            input_minus_mu = (input - self._mu)
            dinput_norm = grad_output * self.gamma
            dsigma = np.sum(dinput_norm*input_minus_mu*(-.5)*self.std_inv**3, axis=0)
            dmu = np.sum(dinput_norm * (-self.std_inv), axis=0) \
                  + dsigma * np.mean(-2 * input_minus_mu, axis=0)
            
            self.grad_gamma = np.sum(grad_output * self._input_norm, axis=0)
            self.grad_beta = np.sum(grad_output, axis=0)
            grad_input = dinput_norm*self.std_inv + dsigma*input_minus_mu/m + dmu/m
        else:
            self.grad_gamma = np.sum(grad_output * self._input_norm, axis=0)
            self.grad_beta = np.sum(grad_output, axis=0)
            grad_input = grad_output * self.gamma * self.std_inv
        return grad_input

    def parameters(self):
        return [self.gamma, self.beta]
    
    def grad_parameters(self):
        return [self.grad_gamma, self.grad_beta]

    def _normalize(self, input, mu, sigma):
        self.std_inv = 1/np.sqrt(sigma + self.eps)
        return (input - mu)*self.std_inv