
# Функции активации
- [Sigmoid](#sigmoid)
- [Tanh](#tanh-гиперболический-тангенс)
- [ReLU](#relu-rectified-linear-unit)
- [Leaky ReLU](#leaky-relu)
- [Softmax](#softmax)
## Sigmoid
[Источник](https://academy.yandex.ru/handbook/ml/article/pervoe-znakomstvo-s-polnosvyaznymi-nejrosetyami)

Исторически одна из первых функций активации. Рассматривалась в том числе и как гладкая аппроксимация порогового правила, эмулирующая активацию естественного нейрона.
$$\sigma(x)=\frac{1}{1+e^{-x}}$$
$$\sigma: \mathbb{R} \to (0, 1)$$
![sigmoid plot](/imgs/sigmoid.jpg)

На практике редко используется внутри сетей, чаще всего в случаях, когда внутри модели решается задача бинарной классификации.

Проблемы:
- На концах функции (значения рядом с 0 и 1) производная практически равна 0, что приводит к затуханию градиентов.
- область значений смещена относительно нуля;
- exp вычислительно дорогая операция (ReLU быстрее в 4-6 раз)

### Backward
$$\frac{d\sigma}{dx}=\frac{exp(-x)}{(1+exp(-x))^2}=\frac{1-1+exp(-x)}{(1+exp(-x))^2}=$$
$$=\frac{1}{1+exp(-x)} - (\frac{1}{1+exp(-x)})^2=\sigma(1-\sigma)$$

### Code
```python
class Sigmoid(Module):
    def forward(self, input):
        self.output = self.__class__._sigmoid(input)
        return self.output

    def backward(self, input, grad_output):
        sigma = self.output
        grad_input = np.multiply(grad_output, sigma*(1 - sigma))
        return grad_input
        
    @staticmethod
    def _sigmoid(x):
        return 1/(1 + np.exp(-x))
```
---
## **Tanh**, гиперболический тангенс
[Источник](https://academy.yandex.ru/handbook/ml/article/pervoe-znakomstvo-s-polnosvyaznymi-nejrosetyami)

Tanh решает проблему несимметричности Sigmoid и также может быть записан через неё $tanh(x)=2 \times \alpha(2x)-1$.
$$\tanh(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}$$
$$\tanh: \mathbb{R} \to (-1, 1)$$

![tanh plot](/imgs/tanh.jpg)

Плюсы:
\+ Имеет симметричную область значений относительно нуля (в отличие от сигмоиды)
\+ Имеет ограниченную область (-1, 1)

Минусы:
\- Проблема затухания градиентов на концах функции (близи значений -1 и 1), там производная почти равна 0
\- требуется вычисление exp, что вычислительно дорого

### Backward
$$\tanh(x)=\frac{\sinh(x)}{\cosh(x)}$$
$$\tanh^{'}(x)=\frac{1}{\cosh^2(x)}$$
Из [основного тождества](https://ru.wikipedia.org/wiki/Гиперболические_функции) $\cosh^2(x)-\sinh^2(x)=1$ имеем:
$$1-tanh^2(x)=\frac{1}{\cosh^2(x)}$$

### Code
```python
class Tanh(Module):
    def forward(self, input):
        self.output = np.tanh(input)
        return self.output

    def backward(self, input, grad_output):
        th = self.output
        grad_input = np.multiply(grad_output, (1 - th*th))
        return grad_input
```
---
## **ReLU**, Rectified linear unit
[Источник](https://academy.yandex.ru/handbook/ml/article/pervoe-znakomstvo-s-polnosvyaznymi-nejrosetyami)

ReLU представляет собой простую кусочно-линейную функцию. Одна из наиболее популярных функций активации. В нуле производная доопределяется нулевым значением.
$$\text{ReLU}(x)=\max(0, x)$$
$$\text{ReLU}: \mathbb{R} \to [0, +\infty)$$

![relu plot](/imgs/relu.jpg\)

Плюсы:
\+ сходится быстро (относительно sigmoid из-за отсутсвие проблемы с затуханием градиентов)
\+ вычислительная простота активции и производной (Прирост в скорости относительно сигмойды в 4-6 раз)
\+ не saturated nonlinear

Минусы:
\- для отрицательных значений производная равна нулю, что может привести к затуханию градиента;
\- область значений является смещённой относительно нуля.

### Backward
Пусть $f(x)=\text{ReLU}(x)$, тогда
$$\frac{df}{dx}=\left\{\begin{matrix}
0, \quad x \leqslant  0
 \\ 1, \quad x > 0
\end{matrix}\right.$$

### Code
```python
class ReLU(Module):
	"""Rectified linear unit. Activation function."""
	
    def forward(self, input):
        self.output = np.maximum(input, 0)
        return self.output

    def backward(self, input, grad_output):
        grad_input = np.multiply(grad_output, input > 0)
        return grad_input
```
---
## Leaky ReLU
[Источник](https://academy.yandex.ru/handbook/ml/article/pervoe-znakomstvo-s-polnosvyaznymi-nejrosetyami)

Модификация ReLU устраняющая проблему смерти градиентов при $x < 0$, тем самым меньше провоцируя затуханием градинета. Гиперпараметр $α$ обеспечивает небольшой уклон слева от нуля, что позволяет получить более симметричную относительно нуля область значений (чаще всего $\alpha = 0.01$).
$$\text{LReLU}(x) = \max(\alpha x, x), \quad  0 < \alpha \ll 1$$
$$\text{LReLU}: \mathbb{R} \to (-\infty, +\infty)$$
![leaky relu plot](/imgs/leaky_relu.jpg)

### Backward
Пусть $f(x)=\text{ReLU}(x)$, тогда
$$\frac{df}{dx}=\left\{\begin{matrix}
    \alpha, \quad x \leqslant  0
 \\ 1, \quad x > 0
\end{matrix}\right.$$

### Code
```python
class LeakyReLU(Module):

    def __init__(self, slope=0.01):
        super().__init__()
        self.slope = slope

    def forward(self, input):
        self.output = (input > 0)*input + (input <= 0)*self.slope*input
        return self.output

    def backward(self, input, grad_output):
        grad_input = np.multiply(
	        grad_output, (input > 0) + (input <= 0)*self.slope
	        )
        return grad_input
```
---
## Softmax
[Источник 1](https://en.wikipedia.org/wiki/Softmax_function), [Источник 2](https://sgugger.github.io/a-simple-neural-net-in-numpy.html)

**Softmax** также известна, как **softargmax** или **normalized exponential function**. Softmax преобразует вектор из $K$ вещественных чисел в вероятностное распределение $K$ возможных исходов. Чаще всего softmax встречается, как активация на последнем слое в задачах многоклассовой классификации.
$$\sigma(z)_i=\frac{e^{z_i}}{\sum^{K}_{j=1}{e^{z_j}}} \quad,i=1,\dots,K$$
$$\sigma:\mathbb{R}^K \to (0,1)^K$$
Сумма элементов выходного вектор равна 1.  Softmax является аппроксимацией функции argmax.

### Backward
Если мы хотим продифференциировать $\sigma_i$ , то мы можем продифференцировать её по $K$ переменным. Пусть $p_i=\sigma(z)_i$
$$\frac{ \partial p_i}{\partial z_i}=\frac{e^{z_i}\sum^K_{j=1}{e^{z_j}} - e^{2z_i}}{(\sum^K_{j=1}{e^{z_j}})^2}=\frac{e^{z_i}}{\sum^K_{j=1}{e^{z_j}}}-\frac{e^{2z_i}}{(\sum^{K}_{j=1}{e^{z_j}})^2}=p_i(1-p_i)$$
для всех $i\neq j$, то:
$$\frac{\partial p_i}{\partial z_j}=-\frac{e^{z_i}e^{z_j}}{(\sum^{K}_{j=1}{e^{z_j}})^2}=-\frac{e^{z_i}}{\sum^{K}_{j=1}{e^{z_j}}}*\frac{e^{z_j}}{\sum^{K}_{j=1}{e^{z_j}}}=-p_ip_j$$
Пусть $l$ - некоторая функция, принимающая на вход выход softmax (например функция потерь). Тогда используя правило дифференцирование сложной функции (chain rule) получаем:
$$\frac{\partial l}{\partial z_i}=\sum_{j=1}^{K}\frac{\partial l}{\partial p_j}\frac{\partial p_j}{\partial z_i}=p_i(1-p_i)\frac{\partial l}{\partial p_i} - \sum_{i \neq j}p_ip_j\frac{\partial l}{\partial p_j}=$$
$$=p_i\frac{\partial l}{\partial p_i}-\sum_{j=1}^{K}{p_ip_j\frac{\partial l}{\partial p_j}}$$

### Code
```python
class Softmax(Module):
    def forward(self, input):
        self.output = self.softmax = self._softmax(input)
        return self.output

    def backward(self, input, grad_output):
        p = self.softmax
        grad_input = p * ( grad_output - (grad_output * p).sum(axis=1)[:, None] )
        return grad_input

    def _softmax(self, x):
        x = np.subtract(x, x.max(axis=1, keepdims=True))
        e_m = np.exp(x)
        sum_e = np.repeat(np.sum(e_m, axis=1), x.shape[-1]).reshape(*e_m.shape)
        return e_m/sum_e
```