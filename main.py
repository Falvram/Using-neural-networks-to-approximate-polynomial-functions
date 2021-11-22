import math

import numpy
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from numpy import asarray, arange
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler


class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, save_best_metric='val_loss', this_max=False):
        self.save_best_metric = save_best_metric
        self.max = this_max
        if this_max:
            self.best = float('-inf')
        else:
            self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs[self.save_best_metric]
        if self.max:
            if metric_value > self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()

        else:
            if metric_value < self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()


def train_predict_model(x, y, m):
    # CONSTANTS
    C = 1  # C > 0
    # m = 3  # Degree of a polynomial
    eps = 0.01  # ε ∈ (0, 1/2)
    D = 1  # the boundaries of the segment in which we perform the approximation
    W = 16  # W ≤ 16 - network width

    # POLYNOMIAL FUNCTION
    # x = asarray([i for i in arange(-1, 1.01, 0.01)])
    # a = asarray([i for i in range(2, 5)])
    # y = asarray([a[0] * i ** 2 + a[1] * i ** 3 + a[2] * i ** 4 for i in x])

    # VARIABLES
    A = 1  # maximum of factor
    L = C * m * (math.log2(A) + math.log2(math.pow(eps, -1)) + m * math.log2(D) + math.log2(m))  # depth of network ≤ L

    # SCALE THE INPUT DATA
    x = x.reshape((len(x), 1))
    y = y.reshape((len(y), 1))
    scale_x = MinMaxScaler()
    x = scale_x.fit_transform(x)
    scale_y = MinMaxScaler()
    y = scale_y.fit_transform(y)

    # NETWORK TRAINING
    model = Sequential()
    model.add(Dense(W, input_dim=1, activation='relu'))
    for i in range(int(L)):
        model.add(Dense(W, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    save_best_model = SaveBestModel()
    model.fit(x, y, validation_data=(x, y), epochs=500, batch_size=10, verbose=0, callbacks=[save_best_model])
    model.set_weights(save_best_model.best_weights)

    # NETWORK PREDICT
    yhat = model.predict(x)

    # PLOT RESULTS
    x_plot = scale_x.inverse_transform(x)
    y_plot = scale_y.inverse_transform(y)
    yhat_plot = scale_y.inverse_transform(yhat)
    # report model error
    print('MSE: %.3f' % mean_squared_error(y_plot, yhat_plot))
    # plot x vs y
    pyplot.scatter(x_plot, y_plot, label='Actual')
    # plot x vs yhat
    pyplot.scatter(x_plot, yhat_plot, label='Predicted')
    pyplot.title('Input (x) versus Output (y)')
    pyplot.xlabel('Input Variable (x)')
    pyplot.ylabel('Output Variable (y)')
    pyplot.legend()
    pyplot.show()
    return x_plot, y_plot, yhat_plot


def evaluate_two_power_n(y, n):
    y_predicted = y
    for i in range(n):
        y = asarray([k ** 2 for k in y])
        x_pl, y_pl, y_predicted = train_predict_model(x, y, 2)
        # plot(x_pl, y_pl, y_predicted)
    return y_predicted


def two_k_ary_brauer(num, k):
    binary = "{0:b}".format(num)
    f = 0
    res = []
    while f != len(binary) - len(binary) % k:
        part = ""
        for i in range(k):
            part += binary[f]
            f += 1
        res.append([int(part, 2), len(binary) - f])
        print(int(part, 2), "* 2 ^", len(binary) - f, end='')
        if f == len(binary) - len(binary) % k:
            print()
        else:
            print(" + ", end='')
    return res


def brauer(func):
    power = 51
    print(power, "= ", end='')
    res_list = two_k_ary_brauer(power, 3)
    print(res_list)
    res = asarray([1 for k in x])
    for i in range(len(res_list)):
        y = asarray([k ** res_list[i][0] for k in x])
        x_pl, y_pl, y_predicted = train_predict_model(x, y, res_list[i][0])
        answ = evaluate_two_power_n(y_predicted, res_list[i][1])
        answ = answ.reshape(1, len(answ))
        answ = answ[0]
        res = numpy.multiply(res, answ)
    y = asarray([i ** 51 for i in x])
    pyplot.scatter(x, y, label='Actual')
    # plot x vs yhat
    pyplot.scatter(x, res, label='Predicted')
    pyplot.title('Input (x) versus Output (y)')
    pyplot.xlabel('Input Variable (x)')
    pyplot.ylabel('Output Variable (y)')
    pyplot.legend()
    pyplot.show()
    return res


x = asarray([i for i in arange(-1, 1.01, 0.01)])
function = "x^51"
print("Approximating function", function)
# BRAUER ALGORITHM
print("Using Brauer algorithm")
brauer(function)
