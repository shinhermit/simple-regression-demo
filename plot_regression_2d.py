"""
Plot data points close to a line in a 2D space.

Used to illustrate the section "Understanding linearity"
of the first article about linear regression.
"""

import numpy
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression


def generate_data_set(size=50, seed_x=123, seed_y=321):
    """
    Generate data points lying close to a line.

    :param size: the desired size of the generate data set.
    :param seed_x: seed for random noise to add to the x coordinates of the generated data
    :param seed_y: seed for random noise to add to the y coordinates of the generated data
    :return: x, y values as column vectors numpy.ndarray
    """
    numpy.random.seed(seed_x)
    x = numpy.linspace(0, size, size) + numpy.random.uniform(-size * 0.1, size * 0.1, size)
    numpy.random.seed(seed_y)
    y = numpy.linspace(0, size, size) + numpy.random.uniform(-size * 0.1, size * 0.1, size)
    return x.reshape(-1, 1), y.reshape(-1, 1)


def partition_data(x, y):
    """
    Parition the data set into 3 clusters by
    sampling 1 third of the data for each of
    the training set, the test set and the validation set.

    :param x: values of the feature, or input data
    :param y: values of the target variable
    :return: slices of the data set as x_train, y_train, x_test, y_test
    """
    size = x.shape[0]
    step = 3
    x_train = x[:size:step]
    y_train = y[:size:step]
    x_test = x[1:size:step]
    y_test = y[1:size:step]
    return x_train, y_train, x_test, y_test


def display_results(x, y, model):
    """
    Plot the results.

    - Plot the data set as a scatter plot
    - Plot the trained model as a plane. We use the coefficients
      a calculated by the model to plot the plane.
    - Plot the generator equation as a plane.

    :param x:the features or inputs of the data set,. x has the shape:
    [[x1],
     [x2]
     ...
     [xn]]
    :param y: the target value, as a 1D ndarray
    :param model: the trained scikit learn model
    data set.
    """
    # plot the data
    pyplot.scatter(x, y,  color='black')
    # plot the trained model as a line
    pyplot.plot(x, model.predict(x),
                color='blue', linewidth=3)


def main():
    x, y = generate_data_set()
    x_train, y_train, x_test, y_test = partition_data(x, y)

    model = LinearRegression()
    model.fit(x_train, y_train)

    display_results(x, y, model)

    pyplot.show()


if __name__ == "__main__":
    main()
