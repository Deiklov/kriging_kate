import numpy as np
import math

np.random.seed(42)
# Функция подсчета градиента
def gradient(y_true: int, y_pred: float, x: np.array) -> np.array:
    """
    y_true - истинное значение ответа для объекта x
    y_pred - значение степени принадлежности объекта x классу 1, предсказанное нашей моделью
    x - вектор признакового описания данного объекта

    На выходе ожидается получить вектор частных производных H по параметрам модели, предсказавшей значение y_pred
    Обратите внимание, что размерность этого градиента должна получиться на единицу больше размерности x засчет своободного коэффициента a0
    """
    dhdw = lambda data: data*((1-y_true)*y_pred-y_true*(1-y_pred))

    dhdwArr = np.vectorize(dhdw)

    return dhdwArr(np.append(x,1))


# Функция обновления весов
def update(alpha: np.array, gradient: np.array, lr: float):
    """
    alpha: текущее приближения вектора параметров модели
    gradient: посчитанный градиент по параметрам модели
    lr: learning rate, множитель перед градиентом в формуле обновления параметров
    """
    alpha_new = alpha-lr*gradient
    return alpha_new


# функция тренировки модели
def train(
    alpha0: np.array, x_train: np.array, y_train: np.array, lr: float, num_epoch: int
):
    """
    alpha0 - начальное приближение параметров модели
    x_train - матрица объект-признак обучающей выборки
    y_train - верные ответы для обучающей выборки
    lr - learning rate, множитель перед градиентом в формуле обновления параметров
    num_epoch - количество эпох обучения, то есть полных 'проходов' через весь датасет
    """
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    alpha = alpha0.copy()
    for epo in range(num_epoch):
        for i, x in enumerate(x_train):
            y_pred=sigmoid(np.sum(np.append(x,1)*alpha))
            grad=gradient(y_train[i],y_pred,x)
            alpha=update(alpha,grad,lr)
    return alpha