"""
Plots the article's example of a 2D regression.
(sectoion "Understanding linearity")
"""

import numpy
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression


def generate_data_set(size=50, seed_x=123, seed_y=321):
    numpy.random.seed(seed_x)
    x = numpy.linspace(0, size, size) + numpy.random.uniform(-size * 0.1, size * 0.1, size)
    numpy.random.seed(seed_y)
    y = numpy.linspace(0, size, size) + numpy.random.uniform(-size * 0.1, size * 0.1, size)
    return x, y


def partition_data(x, y):
    size = x.shape[0]
    partition_size = int(size / 3)
    step = int(size / partition_size)
    x_train = x[:partition_size:step]
    y_train = y[:partition_size:step]
    x_test = x[1:partition_size:step]
    y_test = y[1:partition_size:step]
    return x_train, y_train, x_test, y_test


def display_results(x, y, model):
    # plot the data
    pyplot.scatter(x, y,  color='black')
    # plot the trained model as a line
    pyplot.plot(x, model.predict(x.reshape(-1, 1)),
                color='blue', linewidth=3)


def main():
    x, y = generate_data_set()
    x_train, y_train, x_test, y_test = partition_data(x, y)

    model = LinearRegression()
    model.fit(x_train.reshape(-1, 1),
              y_train.reshape(-1, 1))

    display_results(x, y, model)

    pyplot.show()


if __name__ == "__main__":
    main()
