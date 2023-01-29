# Основные слои нейронной сети
- [Linear](#linear)
- [Batch Normaliztion](#batch-normalization)
- [Dropout](#dropout)
## Linear
**Линейнейный слой** (linear layer) или **полносвясный** (full connected layer) производит линейное преобразование над входными данными $x\rightarrow xW+b$, где $W \in \mathbb{R}^{m\times n}$ – матрица весов, $b \in \mathbb{R}^{n}$ – смещение или bias, $x \in \mathbb{R}^{m}$ – входные данные. 
$$z(x)=xW+b$$
То есть в случае подачи входного батча данных размера $k \times m$ сначало он домножается на матрицу весов $W$ размера $m \times n$ и получается матрица размера $k \times n$  к каждой строке которой прибавляется вектор-столбец $b$ и на выходе получается выходной батч размера $k \times n$.

Веса и баес инициализируются согласно инициализации Xavier.

### Backward
Рассмотрим случай батча размера $k=1$. Пусть есть линейная функция $z(x)=xW+b$ и некоторая гладкая функция $g(x)$, представляющая из себя все последующие за линейным слоём преобразования $g(z(x))$, дифиринцируя её по входным значениям:
$$\frac{\partial g}{\partial x_i}=\sum_{j=1}^{n}\frac{\partial g}{\partial z_j}\frac{\partial z_j}{\partial x_i}=\sum_{j=1}^{n}{\frac{\partial g}{\partial z_j}w_{ij}},\space \space \space i=1,2,...,m$$
Дифференцируем её по весам:
$$\frac{\partial g}{\partial w_{ij}}=\frac{\partial g}{\partial z_j}\frac{\partial z_j}{\partial w_{ij}}=\frac{\partial g}{\partial z_j}x_i,\space \space \space i=1,2,...,m; \space j=1, 2, ..., n$$
Дифференцируем её по смещению:
$$\frac{\partial g}{\partial b_{j}}=\frac{\partial g}{\partial z_j}\frac{\partial z_j}{\partial b_{j}}=\frac{\partial g}{\partial z_j}$$
Запишем в виде градиентов:
$$\nabla_{x}{g}=\nabla_{z}{g} \cdot W^T$$
$$\nabla_{W}{g}=x^T \cdot \nabla_{z}{g}$$
$$\nabla_b{g}=\nabla_z{g}$$
### Code
```python
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
```
---
## Batch Normalization
[Источник](https://arxiv.org/pdf/1502.03167.pdf)

Нормирует вход слоя сети по каждому обучающиму mini-batch(то есть m > 1).

Эффекты слоя:
- решает проблему Internal Covariate Shift*, что **сильно ускоряет сходимость**;
- так же действует в качестве регулиризатора, что позволяет убрать или снизить влияние Dropout;
- позволяет использовать saturated nonlinearties (например Sigmoid);
- позволяет использовать высокий learning rate без риска несходимости и более небрежную иницилизацию весов.

>По поводу того, где ставить ведутся дискуссии, но анализируя мнение в интернете люди ставят после функции активации
>Но нельзя не отметить, что авторы статьи ставят перед функцией активации(хотя далее сообщеет разработчик Keras, что автор статьи сейчас ставит после функции активации)
>[Обсуждение на stackoverflow](https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout) также рассматривается и вопрос куда ставить Dropout.

### Подробнее
Сходимость нейросети ускоряется, если на вход сети подаются нормализованные данные. То же самое верно и для "подсетей" нейросети (1 и более слоёв).

Для преобразования данных внутри слоёв сети данный слой используется два упрощения позволяющих решить задачу нормализации слоёв на всей статистики обучения:
1. Нормализировать каждую scalar input features независимо (var=1, mean=0);
2. Каждый мини-батч выдаёт оценку дисперсии и мат. ожидания для входа.

### Covariate Shift
**Covariate Shift** представляет из себя проблему при которой изменяется распределения входа нейросети, которое приводит к тому, что нужно подстраиваться под новое распределение, что замедляет обучение.

**Internal Covariate Shift** является той же проблемой только в масштабах "подсети" нейросети, то есть проблемой для одного или нескольких слоёв. Так после очередного gradient otimizer step распределение на выходе одного из слоёв может изменится, что заставит последующие подстраиваться.

### Метод
1. Принимаем на вход мини-батч выхода предыдущего слоя размера m>1.
2. Нормализируем все фичи отдельно для выхода предыдущего слоя с $d$ измерениями $x=(x^{(1)}, x^{(2)}, ..., x^{(d)})$.	$$\large\hat{x}^{(k)} = \frac {x^{(k)} − E[x^{(k)}]}{\sqrt{Var[x(k)]}}$$
3. Scale and shift the normalize values, используя trainable параметры $\gamma^{(k)}, \beta^{(k)}$, которые "учатся" восстановливать representation power of the model.
$$\large{\hat{y}^{(k)} = \gamma^{(k)}\hat{x}^{(k)} + \beta^{(k)}}$$
> Note. Нормализация входа слоя может приводить к изменению того что может представлять слой. Например, так сигмоиду можно перевести в линейный режим (значения близкие к нулю). Эту проблему и призвана решить линейная трансформация, представленная выше, которая способна репрезентовать идентичную.

**Алгоритм**:
Рассмотрим мини-батч $B={x_{1...m}}$. Опустим $k$, чтобы рассмотреть алгоритм относительно конкретной фичи.
![algorithm batch norm](imgs/algorithmBatchNorm.png)
>The distributions of values of any $\hat{x}$ has the expected value of 0 and the variance of 1, as long as the elements of each mini-batch are sampled from **the same distribution**,and if we neglect $\epsilon$.

### Training
Производные для backprop. Для infernce мы также собираем экспоненциальное скользящие среднее мат. ожидания и дисперсии, представляющие из себя:
$M=M \times c+v \times (1-c)$, где $M$ скользящее среднее, $0< c < 1$, $v$ новый элемент. Я брал $c=0.9$.

![gradient batch norm](imgs/gradBatchNorm.png)

### Inference
Фиксируем параметры $\gamma, \beta$ и берём мат. ожидание и дисперсию, как скользящее среднее от значений при обучение. 

![inference batch norm](imgs/inferenceBatchNorm.png)

### Code
```python
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
```

### Batch-Normalized Convolution Networks
Нормализируем также по каждому значенею матрицы feature maps, но обучаем $\gamma, \beta$ для каждого feature maps, а не для каждого его значения. 

---
## Dropout
[Источник](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)

P.S. нейроны - можно читать как units или выходы предыдущего слоя.

Метод регуляризации главная идея которого заключается в отключение части нейронов с некоторой вероятностью $p$. Метод позволяет уменьшить переобучение и коадаптацию нейронов.

![dropout viz](imgs/dropout_visualisation.png)

Обычно оптимальное значение $p=0.5$.

### Подробнее
По сути, вместо одной большой сети мы обучаем $2^n$  возможных подсетей с общими весами. Во время инференса сложно брать среднее предсказания по экспоненциально большому количству подсетей, поэтому используется следующая аппроксимация. Во время инференса мы используем сеть полностью (без dropout) и домножаем веса на $1-p$ (или домножать не отключенные веса на $\frac{1}{1-p}$ во время обучения). Это гарантирует нам, что мат. ожидание выхода нейросети во время обучения совпадёт с мат. ожиданием во время инференса.

![inference with dropout](imgs/inference_with_dropout.png)

Также не могу не отметить, то как применяется dropout слой в нейронной сети. Пусть dropout является $l$-ым слоем, тогда он применяется к выходу предыдущего слоя $l-1$ , что влияет на веса слоя $l+1$, но не на его bias!

![nn and dropout](imgs/nn_and_dropout.png)

P.S. Мои пометки довольно условны, но надеюсь понятны.

### Forward
Мы зануляем часть входа предыдущего слоя тем самым "отключая" некоторые нейроны. (В случае, если мы не хотим во время инференса домножать на $1-p$, то домножаем не занулённые веса на $\frac{1}{1-p}$)

### Backward
Мы сохраняем маску (если домножали на $\frac{1}{1-p}$, то сохраняем это в маске), по которой зануляли вход при forward и применяем её также для градиентов при backward. 

### Code
```python
class Dropout(Module):
	def __init__(self, p=0.5):
		super().__init__()
		self.p = p
		self.mask = None

	 def forward(self, input):
		if self._train:
			p_save = 1 - self.p
			self.mask = np.random.binomial(
				1, p=p_save, size=input.shape)/p_save
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
```

### Советы
**Выбор гиперпараметра $p$**, где $p$ – это вероятность того, что нейрон исчезнет из сети. Обычно оптимальное значение $p \in [0.2, 0.5]$. Для вещественного входного слоя, вроде speech frame или image patches подходит $p=0.2$. Для скрытого слоя выбор $p$ зависит от количества нейронов в слое.

**Настройска размер сети**. Пусть в каком-то скрытом слое нашей сети $n$ нейронов и они исчезают с некоторой вероятностью $p$, тогда наша сеть будь представлена $(1-p)n$ нейронами после дропаута. То есть если для некоторого слоя стандартной нейросети оптимально $n$ нейронов для решаемой задачи, то для хорошей дропаут нейросети это количество будет по крайне мере $n/(1-p)$. (Эвристика предложенная авторами статьи для full-connected и convolution сетей)  

Авторы статьи отмечают, что dropout нейросети особенно хорошо работают в купе с:
- высоким momentum. Авторы указывают, что 0.95-0.99 работает достаточно хорошо, также можно использовать обычный SGD с learning rate в 10-100 раз больше чем обычный. Это позволяет значительно ускорить обучение.
- Max-norm Regularization. Высокий learning rate/momentum приводит к сильному увелечению значения весов. Для предотвращение этого можно использовать max-norm регулиризацию. Оптимальное значение $c$ в промежутке от 3 до 4.
- большим learning decay.
