# Функции потерь
- [MSE](#mean-squared-error)
- [Cross Entropy](#cross-entropy)
## Mean squared error
[Источник](https://en.wikipedia.org/wiki/Mean_squared_error)

Среднеквадратичная ошибка - это функция потерь, часто используемая для задач регрессии. Пусть $N$ – это размер батча, $y$ – это целевое значение, а $p$ – это предсказание модели, тогда: 
$$MSE=\frac{1}{N}\sum_{i=1}^{N}(y_i-p_i)^2$$

### Backward
Пусть $l(y, p)=MSE(y, p)$, тогда:
$$\frac{\partial l}{\partial p}=\frac{2}{N}\sum_i(y_i-p_i)$$

### Code
```python
class MSE(Criterion):
    def forward(self, input, target):
        batch_size = input.shape[0]
        self.output = np.sum(np.power(input - target, 2)) / batch_size
        return self.output

    def backward(self, input, target):
        self.grad_output  = (input - target) * 2 / input.shape[0]
        return self.grad_output
```
---
## Cross entropy
[Источник](https://deepnotes.io/softmax-crossentropy#derivative-of-softmax)

Cross entropy определяет растояние между двумя распределениями: реальным распределением $y$ и предсказанным $p$. Данная функция потерь в частности, используется для решения задачи классификации на $C$ классов. 
$$H(y, p)=-\sum_{i=1}^Cy_i\log{p_i}$$

### Backward
$$l=-\sum_{i=1}^Cy_i\log{p_i}$$
$$\frac{\partial l}{\partial p_i}=-y_i\frac{1}{p_i}$$
Чаще всего в задачах классификации на несколько классах в качестве последнего слоя нейросети используется Softmax. Воспользовавшись производной посчитанной в заметке Softmax (Размер вектора $z$ в случае когда Softmax стоит перед кросс энтропией равен количеству классов $K=C$).
$$\frac{\partial l}{\partial z_i}=\sum_{j=1}^{C}\frac{\partial l}{\partial p_j}\frac{\partial p_j}{\partial z_i}=p_i\frac{\partial l}{\partial p_i}-\sum_{j=1}^{C}{p_ip_j\frac{\partial l}{\partial p_j}}=$$
$$=-p_iy_i\frac{1}{p_i}+\sum_{j=1}^{C}{y_ip_ip_j\frac{1}{p_j}}=-y_i+p_i\sum_{j=1}^{C}{y_j}$$
В случае задачи многлассовой классификации мы имеем one-hot вектор $y$ класса $i$ такой что $y_i=1$, а $y_j=0, \quad i \neq j$, то есть $\sum y_i=1$.
$$\frac{\partial l}{\partial z_i}=p_i-y_i$$
Используя одним слоем Softmax+CrossEntropy позволяет сильно уменьшить количество вычислений.

### Code
```python
class CrossEntropy(Criterion, Softmax):
    """Тоже самое, что и Sofrmax + NLL только быстрее.
    Соответсвенно принимает не распределение вероятностей, a логиты (выход Linear).
    """
    def forward(self, input, target):
        batch_size = input.shape[0]
        self.prob = self._softmax(input)
        # чтобы нигде не было взятий логарифма от нуля:
        eps = 1e-9
        prob_clamp = np.clip(self.prob, eps, 1 - eps)
        self.output = np.sum(-np.log(prob_clamp) * target) / batch_size
        return self.output

    def backward(self, input, target)
        batch_size = input.shape[0]
        eps = 1e-9
        prob_clamp = np.clip(self.prob, eps, 1 - eps)
        self.grad_output = (prob_clamp - target) / batch_size
        return self.grad_output
```

### Примечание
- [Связь с методом максимального правдоподобия](https://leimao.github.io/blog/Cross-Entropy-KL-Divergence-MLE/)
