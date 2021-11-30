import math
import random

import sympy
import time
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


def train_predict_model(x, y, m, absolute_power):
    # CONSTANTS
    C = 1  # C > 0
    # m = 1  # Degree of a polynomial
    eps = 0.01  # ε ∈ (0, 1/2)
    D = 1  # the boundaries of the segment in which we perform the approximation
    W = 16  # W ≤ 16 - network width

    # POLYNOMIAL FUNCTION
    # x = asarray([i for i in arange(-1, 1.01, 0.01)])
    # a = asarray([i for i in range(2, 5)])
    # y = asarray([a[0] * i ** 2 + a[1] * i ** 3 + a[2] * i ** 4 for i in x])

    # VARIABLES
    # A = 1  # maximum of factor
    if m == 0:
        m = 1
    L = C * m * (math.log2(A) + math.log2(math.pow(eps, -1)) + m * math.log2(D) + math.log2(m))  # depth of network ≤ L
    if L > 997:
        L = 900
    print("A, m, Depth:", A, m, L)

    # SCALE THE INPUT DATA
    # var_x = asarray([i for i in arange(-3, 3.01, 0.01)])
    # var_y = asarray([i ** m__ for i in var_x])
    # var_x = numpy.append(numpy.random.power(absolute_power, 1000), -numpy.random.power(absolute_power, 1000))
    var_x = []
    var_x = numpy.append(var_x, x)
    # var_x = []
    # var_x = numpy.append(var_x, numpy.random.rand(1, 10000))
    # var_x = numpy.append(var_x, -numpy.random.rand(1, 10000))
    # var_x = numpy.append(var_x, [random.uniform(0.875, 1) for i in range(50000)])
    # var_x = numpy.append(var_x, [-random.uniform(-1, -0.875) for i in range(50000)])
    # var_x = numpy.append(var_x, [-random.uniform(-1, -0.998) for i in range(5000)])
    # var_x = numpy.append(var_x, [random.uniform(1, 0.998) for i in range(5000)])
    var_x = var_x.reshape(len(var_x), 1)
    var_y = asarray([i ** absolute_power for i in var_x])

    # scale_var_x = MinMaxScaler()
    # scale_var_y = MinMaxScaler()
    # var_x = scale_var_x.fit_transform(var_x)
    # var_y = scale_var_y.fit_transform(var_y)
    #
    x = x.reshape((len(x), 1))
    y = y.reshape((len(y), 1))
    # scale_x = MinMaxScaler()
    # scale_y = MinMaxScaler()
    # x = scale_x.fit_transform(x)
    # y = scale_y.fit_transform(y)

    # NETWORK TRAINING
    model = Sequential()
    model.add(Dense(W, input_dim=1, activation='relu'))
    for i in range(int(L)):
        model.add(Dense(W, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='sgd', metrics=['binary_accuracy'])
    save_best_model = SaveBestModel()
    history = model.fit(var_x, var_y, validation_split=0.2, epochs=5, batch_size=10, verbose=0,
                        callbacks=[save_best_model])
    model.set_weights(save_best_model.best_weights)
    print(history.history.keys())
    show_acc_loss(history)

    # NETWORK PREDICT
    yhat = model.predict(x)

    # PLOT RESULTS
    # x = scale_x.inverse_transform(x)
    # y = scale_y.inverse_transform(y)
    # yhat = scale_y.inverse_transform(yhat)
    # report model error
    print('MSE: %.3f' % mean_squared_error(y, yhat))
    plot(x, y, yhat)
    yhat = yhat.reshape(1, len(yhat))
    return yhat[0], absolute_power


def plot(actual_x, actual_y, predicted_y):
    # plot x vs y
    pyplot.scatter(actual_x, actual_y, label='Actual')
    # plot x vs yhat
    pyplot.scatter(actual_x, predicted_y, label='Predicted')
    pyplot.title('Input (x) versus Output (y)')
    pyplot.xlabel('Input Variable (x)')
    pyplot.ylabel('Output Variable (y)')
    pyplot.legend()
    pyplot.show()


def show_acc_loss(history):
    pyplot.plot(history.history['binary_accuracy'])
    pyplot.plot(history.history['val_binary_accuracy'])
    pyplot.title('model accuracy')
    pyplot.ylabel('accuracy')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'test'], loc='upper left')
    pyplot.show()
    # summarize history for loss
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'test'], loc='upper left')
    pyplot.show()


def evaluate_two_power_n(y, n, absolute_power):
    y_predicted = y
    for i in range(n):
        y = asarray([k ** 2 for k in y])
        y_predicted, absolute_power = train_predict_model(x, y, 2, absolute_power * 2)
    return y_predicted, absolute_power


def n_in_radix_two_k(num, k):
    binary = "{0:b}".format(num)
    f = 0
    res = []
    if num < 2 ** k:
        return [[num, 0]]
    while f != len(binary) - len(binary) % k:
        part = ""
        for i in range(k):
            part += binary[f]
            f += 1
        res.append([int(part, 2), len(binary) - f])
    return res


def brauer(coeff, power):
    k = 3
    res_list = n_in_radix_two_k(power, k)
    print(res_list)
    res = asarray([1 for k in x])
    absolute_power = 1
    for i in range(len(res_list)):
        y = asarray([k ** res_list[i][0] for k in x])
        y_predicted, absolute_power = train_predict_model(x, y, res_list[i][0], res_list[i][0])
        answ, absolute_power = evaluate_two_power_n(y_predicted, res_list[i][1], absolute_power)
        # answ = answ.reshape(1, len(answ))
        answ = answ[0]
        res = numpy.multiply(res, answ)
    res, absolute_power = train_predict_model(x, asarray([coeff * i for i in res]), 1, absolute_power)
    y = asarray([coeff * i ** power for i in x])
    plot(x, y, res)
    return res.reshape(1, len(res))


def yao_s(coeff, power):
    k = 3
    factorization = []
    res = n_in_radix_two_k(power, k)
    for i in range(1, 2 ** k - 1):
        tmp = yao_s_d(i, res)
        if tmp[1]:
            factorization.append(tmp)
    answ = [1 for i in x]
    print(factorization)
    absolute_power = 1
    for i in range(len(factorization)):
        y = asarray([i for i in x])
        term = asarray([1 for i in x])
        for k in range(len(factorization[i][1])):
            temp, absolute_power = evaluate_two_power_n(y, factorization[i][1][k], 1)
            term = numpy.multiply(term, temp)
        term, absolute_power = train_predict_model(x, asarray([j ** factorization[i][0] for j in term]),
                                                   factorization[i][0], absolute_power * factorization[i][0])
        answ = numpy.multiply(answ, term)
    answ, absolute_power = train_predict_model(x, asarray([coeff * j for j in answ]), 1, absolute_power)
    return answ


def yao_s_d(n, res):
    answ = []
    for i in range(len(res)):
        if n == res[i][0]:
            answ.append(res[i][1])
    return [n, answ]


def parse_function(function):
    global A
    poly = sympy.polys.polytools.poly_from_expr(function)[0]
    A = max(poly.coeffs())
    return poly.coeffs(), poly.monoms(), poly


def approximate_function(function):
    print("Approximating function", function)
    coeffs, monoms, poly = parse_function(function)
    brauer_res = asarray([0 for i in x])
    yao_s_res = asarray([0 for i in x])
    actual_res = asarray([0 for i in x])
    for i in range(len(coeffs)):
        brauer_res = numpy.add(brauer_res, brauer(coeffs[i], monoms[i][0]))
        yao_s_res = numpy.add(yao_s_res, yao_s(coeffs[i], monoms[i][0]))
        actual_res = numpy.add(actual_res, asarray([coeffs[i] * k ** monoms[i][0] for k in x]))
    predicted_y, c = train_predict_model(x, actual_res, poly.degree(), poly.degree())
    print("Comparing using Brauer predicted function with actual function")
    plot(x, actual_res, brauer_res)
    print("Comparing using Yao's predicted function with actual function")
    plot(x, actual_res, yao_s_res)
    print("Comparing using Standard predicted function with actual function")
    plot(x, actual_res, predicted_y)


def test_two_n_numbers(n):
    y = asarray([i for i in x])
    power = 2 ** n
    print("Time")
    start_time = time.time()
    evaluate_two_power_n(y, n, 1)
    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    train_predict_model(x, [i ** power for i in x], power, power)
    print("--- %s seconds ---" % (time.time() - start_time))


x = asarray([i for i in arange(-1, 1.001, 0.001)])
func = "15 * x ^ 51 + 9 * x ^ 2 + 34"
# approximate_function(func)
test_two_n_numbers(5)
