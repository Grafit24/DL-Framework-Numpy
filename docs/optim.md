# Оптимизаторы
- [SGD](#gradient-descent)
- [Momentum](#momentum)
- [RMSprop](#rmsprop)
- [Adam](#adam)
- [NAdam](#nadam)

## Gradient Descent
[Источник](https://ruder.io/optimizing-gradient-descent/)
### SGD mini-batch
Оптимизация весов модели по градиентам, полученных с loss function относительно весов $θ$, с некоторым learning rate $η$, по её mini-batch'ам, а не конкретным примерам или всей выборки. Для обновление параметров мы усредняем по mini-batch градиенты.
$θ=θ−η⋅∇_θJ(θ;x^{(i:i+n)};y^{(i:i+n)})$

### Vanila GD
Оптимизация по всей выборки. Для обновление параметров мы усредняем градиенты.
$θ=θ−η⋅∇_θJ(θ)$

### SGD
Оптимизация по одному примеру.
$θ=θ−η⋅∇_θJ(θ;x^{(i)};y^{(i)})$


### Проблемы классического подхода
- выбор learning rate;
- отсутствие регулировки learning rate в течение обучения;
- одинаковый learning rate для данных разной частоты;
- попадание  в suboptimal minimum.

---
## Momentum
[Источник](https://ruder.io/optimizing-gradient-descent/)
### Описание

![momentum](/imgs/momentum.png)

Моментум — это метод, который позволяет ускорить SGD и погасить колебания. Методу удаётся это сделать за счёт добавления вектора обновления c предыдущего шага, умноженного на коэффициент $γ$.
$$v_t=γv_{t-1}+η∇_θJ(θ)$$
$$θ=θ−v_t$$
### Интуитивно
Моментум делает наш вектор градиент более похожем на мяч на который теперь действует сила притяжения, из-за чего он скатывается по сколону быстрее и быстрее. Бесконечно сохранять скорость ему не даёт сила сопротивления "воздуха" γ (сохраняемая энергия) при кажом степе. Когда же он перескакивает низ склона залетая на противоположный склон знак градиента меняется и скорость постепенно замедляется после чего, направление меняется и он катится снова к низу. Так он итеративно и доходит до минимума.

---
## RMSprop

[Источник](https://ruder.io/optimizing-gradient-descent/)

Метод пытается решить проблему колебаний, как и Momentum, но заходя с другой стороны RMSprop вычисляет learning rate для каждого параметра отдельно, как Adagrad.
1. Метод позволяющий решить проблему Adagrad с радикально уменьшающимися learning rates;
2. Реализация Rprop для mini-batch.

$$E[g^2]_t = 0.9E[g^2]_{t-1} + 0.1g_t^2$$
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}}g_t$$

---
## Adam

[Источник](https://ruder.io/optimizing-gradient-descent/)

### Описание
Adaptive Moment Estimation - adaptive learning rate метод. В дополнение к сохранению экспоненциального среднего квадратов градиентов $v_t$ как Adadelta и RMSprop, также сохраняет экспоненциальное среднее предыдущих градиентов $m_t$, как Momentum.
$\large m_t = \beta_1m_{t-1} + (1-\beta_1)g_t$
$\large u_t = \beta_2u_{t-1} + (1-\beta_2)g_t^2$

$m_t$ и $u_t$ сдвинуты к нулю, чтобы продействовать этому вычисляют bias-corrected estimates:
$\large \hat{m}_t=\frac{m_t}{1-\beta_1^t}$
$\large \hat{u}_t=\frac{u_t}{1-\beta_2^t}$

$$\large \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{u}_t} + \epsilon} \hat{m}_t$$
## Интутивно
Если Momentum шар, то Adam это тяжёлый шар с сопротивлением, который будет предпочитать минимум на поверхности функции потерь.

---
## NAdam
### Описание
Как Adam только, если Adam=Momentum+RMSprop, то Nadam=NAG+RMSprop.
$$\theta_{t+1} = \theta_t − \frac{\eta}{\sqrt{\hat{u}_t}+\epsilon}  (\beta_1 \hat{m}_t +\frac{(1−\beta_1)g_t}{1−\beta_1^t})$$