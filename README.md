DL-Framework-Numpy
----
Фреймворк глубоко обучения на Numpy, написанный с целью изучения того, как все работает под "капотом". Вместе с фреймворком были написаны конспекты по каждому слою, функции потерь и оптимизатору. Также был написан [пример решения с его помощью задачи распознования рукописных цифр MNIST](examples/mnist.ipynb).

Во многом фреймворк вдохновлялся [pytorch](https://pytorch.org).

Далее перечислено, что было реализовано.

[Cлои](docs/layers.md):
- Linear
- Batch Normalization
- Dropout

[Функции активации](docs/activations.md):
- Sigmoid
- Tanh
- ReLU
- Leaky ReLU
- Softmax
- Logsoftmax

[Функции потерь](docs/loss.md):
- MSE
- NLL
- Cross Entropy

[Оптимизаторы](docs/optim.md):
- SGD
- Momentum
- RMSprop
- Adam
- Nadam

