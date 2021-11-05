import math

from sklearn.metrics import mean_squared_error
from numpy import asarray, arange
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler

# CONSTANTS
C = 1  # C > 0
m = 3  # Degree of a polynomial
eps = 0.01  # ε ∈ (0, 1/2)
D = 1  # the boundaries of the segment in which we perform the approximation
W = 16  # W ≤ 16 - network width

# POLYNOMIAL FUNCTION
x = asarray([i for i in arange(-1, 1.01, 0.01)])
a = asarray([i for i in range(2, 5)])
y = asarray([a[0] * i ** 2 + a[1] * i ** 3 + a[2] * i ** 4 for i in x])

# VARIABLES
A = max(a)  # maximum of factor
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
model.fit(x, y, epochs=500, batch_size=10, verbose=0)

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
