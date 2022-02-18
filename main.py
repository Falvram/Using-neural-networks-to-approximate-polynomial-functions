import math
import random
import sys

import sympy
import time
import numpy
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from numpy import asarray, arange, polyfit, poly1d
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam, Ftrl


class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, save_best_metric='val_loss', this_max=False):
        self.best_weights = None
        self.save_best_metric = save_best_metric
        self.max = this_max
        if this_max:
            self.best = float('-inf')
        else:
            self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs[self.save_best_metric]
        global best_loss
        if self.max:
            if metric_value > self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()
        else:
            if metric_value < self.best:
                best_loss = metric_value
                self.best = metric_value
                self.best_weights = self.model.get_weights()


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        val = optimizer._decayed_lr(var_dtype=tf.float32)
        tf.print(val, output_stream=sys.stdout)
        return val

    return lr


def train_predict_model(x, y, m, power_or_poly):
    # CONSTANTS
    C = 1  # C > 0
    eps = 0.01  # ε ∈ (0, 1/2)
    D = 1  # the boundaries of the segment in which we perform the approximation
    W = 16  # W ≤ 16 - network width

    # VARIABLES
    if m == 0:
        m = 1
    Lc = int(
        C * m * (math.log2(A) + math.log2(math.pow(eps, -1)) + m * math.log2(D) + math.log2(m)))  # depth of network ≤ L
    if Lc > 30:
        L = 30
    else:
        L = Lc
    epochs = int(math.log2(Lc) * Lc) + 200

    # TRAINING DATA
    var_x = []
    var_x = numpy.append(var_x, numpy.random.rand(1, 4096))
    var_x = numpy.append(var_x, -numpy.random.rand(1, 4096))
    numpy.random.shuffle(var_x)
    if isinstance(power_or_poly, sympy.polys.polytools.Poly):
        var_y = asarray([0 for i in var_x])
        for i in range(len(power_or_poly.coeffs())):
            var_y = numpy.add(var_y, asarray([power_or_poly.coeffs()[i] * k **
                                              power_or_poly.monoms()[i][0] for k in var_x]))
        var_x = var_x.reshape(len(var_x), 1)
        var_y = var_y.reshape(len(var_y), 1)
    else:
        if power_or_poly % 2 == 1:
            epochs = 500
        var_x = var_x.reshape(len(var_x), 1)
        var_y = asarray([i ** power_or_poly for i in var_x])

    # SCALE THE INPUT DATA
    scale_var_x = MinMaxScaler()
    scale_var_y = MinMaxScaler()
    var_x = scale_var_x.fit_transform(var_x)
    var_y = scale_var_y.fit_transform(var_y)
    x = x.reshape((len(x), 1))
    y = y.reshape((len(y), 1))
    scale_x = MinMaxScaler()
    scale_y = MinMaxScaler()
    x = scale_x.fit_transform(x)
    y = scale_y.fit_transform(y)

    # NETWORK TRAINING
    model = Sequential()
    model.add(Dense(W, input_dim=1, activation='relu'))
    for i in range(L - 1):
        model.add(Dense(W, activation='relu'))
    model.add(Dense(1))
    learning_rate_fn_4_5 = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=0.0001,
        decay_steps=10000,
        end_learning_rate=0.00001,
        power=0.5)
    learning_rate_fn_5_6 = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=0.00001,
        decay_steps=10000,
        end_learning_rate=0.000001,
        power=0.5)

    model.compile(loss='mse',
                  optimizer=RMSprop())
    save_best_model = SaveBestModel()
    history = model.fit(var_x[0:512], var_y[0:512],
                        validation_split=0.2,
                        epochs=epochs * 5,
                        batch_size=131072,
                        verbose=0,
                        callbacks=[save_best_model])
    model.set_weights(save_best_model.best_weights)
    show_loss(history)

    model.compile(loss='mse',
                  optimizer=Adamax(learning_rate=learning_rate_fn_4_5))
    save_best_model = SaveBestModel()
    history = model.fit(var_x[0:4096], var_y[0:4096],
                        validation_split=0.2,
                        epochs=epochs * 3,
                        batch_size=131072,
                        verbose=0,
                        callbacks=[save_best_model])
    model.set_weights(save_best_model.best_weights)
    show_loss(history)

    model.compile(loss='mse',
                  optimizer=Adamax(learning_rate=learning_rate_fn_5_6))
    save_best_model = SaveBestModel()
    history = model.fit(var_x, var_y,
                        validation_split=0.2,
                        epochs=epochs * 3,
                        batch_size=131072,
                        verbose=0,
                        callbacks=[save_best_model])
    model.set_weights(save_best_model.best_weights)
    show_loss(history)

    # NETWORK PREDICT
    yhat = model.predict(x)

    # PLOT RESULTS
    x = scale_x.inverse_transform(x)
    y = scale_y.inverse_transform(y)
    yhat = scale_y.inverse_transform(yhat)
    # report model error
    print('MAE: %.10f' % mean_absolute_error(y, yhat))
    plot(x, y, yhat)
    yhat = yhat.reshape(1, len(yhat))
    return yhat[0]


def plot(actual_x, actual_y, predicted_y):
    # plot x vs y
    pyplot.scatter(actual_x, actual_y, label='Actual')
    # plot x vs yhat
    pyplot.scatter(actual_x, predicted_y, label='Predicted')
    pyplot.title(title_alg + title_func)
    pyplot.xlabel('Input Variable (x)')
    pyplot.ylabel('Output Variable (y)')
    pyplot.legend()
    pyplot.show()
    print('MAE: %.10f' % mean_absolute_error(actual_y, predicted_y))


def show_loss(history):
    # summarize history for loss
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'test'], loc='upper left')
    pyplot.show()


def evaluate_two_power_n(y, n, absolute_power):
    global title_func
    title_func = ""
    y_predicted = y
    for i in range(n):
        absolute_power *= 2
        title_func = "y = x^%d" % absolute_power
        y = asarray([k ** 2 for k in y])
        y_predicted = train_predict_model(x, y, 2, absolute_power)
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
    global title_alg, title_func
    title_alg = "Brauer algorithm, "
    k = 3
    res_list = n_in_radix_two_k(power, k)
    print(res_list)
    res = asarray([1 for k in x])
    if power == 0:
        res = asarray([coeff * i for i in res])
        y = asarray([coeff * i ** power for i in x])
        title_func = "y = %d * x ^%d" % (coeff, power)
        plot(x, y, res)
        return res.reshape(1, len(res))
    for i in range(len(res_list)):
        absolute_power = res_list[i][0]
        title_func = "y = x^%d" % res_list[i][0]
        y = asarray([k ** absolute_power for k in x])
        y_predicted = train_predict_model(x, y, absolute_power, absolute_power)
        answ, absolute_power = evaluate_two_power_n(y_predicted, res_list[i][1], absolute_power)
        res = numpy.multiply(res, answ)
    res = asarray([coeff * i for i in res])
    y = asarray([coeff * i ** power for i in x])
    title_func = "y = %d * x^%d" % (coeff, power)
    plot(x, y, res)
    return res.reshape(1, len(res))


def yao_s(coeff, power):
    global title_alg, title_func
    title_alg = "Yao's algorithm, "
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
        title_func = ""
        title_func = "y = x^%d" % int(absolute_power * factorization[i][0])
        term = train_predict_model(x, asarray([j ** factorization[i][0] for j in term]),
                                   factorization[i][0], absolute_power * factorization[i][0])
        answ = numpy.multiply(answ, term)
    answ = asarray([coeff * j for j in answ])
    y = asarray([coeff * i ** power for i in x])
    title_func = "y = %d * x^%d" % (coeff, power)
    plot(x, y, answ)
    return answ


def yao_s_d(n, res):
    answ = []
    for i in range(len(res)):
        if n == res[i][0]:
            answ.append(res[i][1])
    return [n, answ]


def least_squares(x, y, deg):
    x = numpy.array(x, dtype='float')
    y = numpy.array(y, dtype='float')
    coeffs = numpy.polyfit(x, y, deg)
    f = poly1d(coeffs)
    plot(x, y, f(x))


def parse_function(function):
    global A
    poly = sympy.polys.polytools.poly_from_expr(function)[0]
    A = max(poly.coeffs())
    return poly.coeffs(), poly.monoms(), poly


def approximate_function(function):
    global title_alg, title_func
    print("Approximating function", function)
    coeffs, monoms, poly = parse_function(function)
    brauer_res = asarray([0 for i in x])
    yao_s_res = asarray([0 for i in x])
    actual_res = asarray([0 for i in x])
    brauer_time = 0
    yao_s_time = 0
    standard_time = 0
    for i in range(len(coeffs)):
        start_time = time.time()
        brauer_res = numpy.add(brauer_res, brauer(coeffs[i], monoms[i][0]))
        brauer_time += time.time() - start_time
        start_time = time.time()
        yao_s_res = numpy.add(yao_s_res, yao_s(coeffs[i], monoms[i][0]))
        yao_s_time += time.time() - start_time
        actual_res = numpy.add(actual_res, asarray([coeffs[i] * k ** monoms[i][0] for k in x]))
    title_alg = "Standard algorithm, "
    title_func = function
    start_time = time.time()
    predicted_y = train_predict_model(x, actual_res, poly.degree(), poly)
    standard_time += time.time() - start_time
    print("Comparing using Brauer predicted function with actual function, time = %s seconds" % brauer_time)
    title_alg = "Brauer algorithm, "
    title_func = function
    plot(x, actual_res, brauer_res)
    print("Comparing using Yao's predicted function with actual function, time = %s seconds" % yao_s_time)
    title_alg = "Yao's algorithm, "
    plot(x, actual_res, yao_s_res)
    print("Comparing using Standard predicted function with actual function, time = %s seconds" % standard_time)
    title_alg = "Standard algorithm, y = "
    title_func = function
    plot(x, actual_res, predicted_y)


def test_two_n_numbers(n):
    global A
    global title_alg, title_func
    A = 1
    y = asarray([i for i in x])
    power = 2 ** n
    print("Time")
    title_alg = "Replacement algorithm, "
    start_time = time.time()
    evaluate_two_power_n(y, n, 1)
    print("--- %s seconds ---" % (time.time() - start_time))
    title_alg = "Standard algorithm, "
    start_time = time.time()
    train_predict_model(x, asarray([i ** power for i in x]), power, power)
    print("--- %s seconds ---" % (time.time() - start_time))


x = asarray([i for i in arange(-1, 1.00001, 0.00001)])
title_alg = ""
title_func = ""
A = 1
best_loss = 0
func = "15 * x ^ 51 + 9 * x ^ 2 + 34"
# approximate_function(func)
test_two_n_numbers(6)

