"""
Plot data points lying close to a plane in a 3D space.

Used to illustrate the section "Understanding linearity"
of the first article about linear regression.
"""
import numpy
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression

from equations import PlaneEquation
from utils import legend_workaround


def generate_data(equation: PlaneEquation, size=50, seed_x=123, seed_y=321, seed_z=1885):
    """
    Generate data points lying close to a plane.

    :param equation: parametric equation of a plane.
    :param size: the desired size of the generate data set.
    :param seed_x: seed for random noise to add to the x coordinates of the generated data
    :param seed_y: seed for random noise to add to the y coordinates of the generated data
    :param seed_z: seed for random noise to add to the z coordinates of the generated data
    :return: x, y, z values as numpy.ndarray
    """
    x, y, z = equation(t=numpy.linspace(0, size, size), s=numpy.linspace(0, size, size))

    numpy.random.seed(seed_x)
    x += numpy.random.uniform(-size * 0.2, size * 0.2, size)
    numpy.random.seed(seed_y)
    y += numpy.random.uniform(-size * 2, size * 2, size)
    numpy.random.seed(seed_z)
    z += numpy.random.uniform(-size * 0.2, size * 0.2, size)
    return x, y, z


def get_training_data(x: numpy.ndarray,
                      y: numpy.ndarray,
                      percent=20) -> (numpy.ndarray, numpy.ndarray):
    """
    Sample the data set at regular steps so as to get a
    part of the data to use as the training set.

    :param x: the features or inputs of the data set,. x has the shape:
    [[x11, x12],
     [x21, x22]
     ...
     [xn1, xn2]]
    :param y: the outputs from the training set. y is just a flat array.
    :param percent: percent of the data set to use for training.
    :return: x_train, y_train as slices of the data set.
    """
    size = x.shape[0]
    sample_size = int(size * percent / 100)
    step = int(size / sample_size)
    return x[:size:step], y[:size:step]


def display_results(x1: numpy.ndarray,
                    x2: numpy.ndarray,
                    y: numpy.ndarray,
                    model: LinearRegression,
                    generator: PlaneEquation=None):
    """
    Plot the results.

    - Plot the data set as a scatter plot
    - Plot the trained model as a plane. We use the coefficients
      a calculated by the model to plot the plane.
    - Plot the generator equation as a plane.

    :param x1: the first feature, as a 1D ndarray
    :param x2: the second feature, as a 1D ndarray
    :param y: the target value, as a 1D ndarray
    :param model: the trained scikit learn model
    :param generator: the plane equation used to generate the
    data set.
    """

    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(azim=25, elev=30)
    
    # plot the data
    ax.scatter(x1, x2, y, c='r', label="Data")

    # for planes
    size=50
    t, s = numpy.meshgrid(numpy.array([0, size]), numpy.array([0, size]))

    # plot the trained model
    equation = PlaneEquation.from_general_form(a=model.coef_[0], b=model.coef_[1],
                                               c=-1, d=model.intercept_)
    x, y, z = equation(t, s)
    surf = ax.plot_surface(x, y, z,
                           alpha=0.5,
                           linewidth=0,
                           antialiased=False,
                           label="Trained model")
    legend_workaround(surf)

    # plot the data generator equation
    if generator is not None:
        x, y, z = generator(t, s)
        surf = ax.plot_surface(x, y, z,
                               color='y',
                               alpha=0.5,
                               linewidth=0,
                               antialiased=False,
                               label="Data generator's equation")
        legend_workaround(surf)

    ax.legend()


def main():
    equation = PlaneEquation.from_general_form(a=10, b=2, c=-10, d=5)
    x1, x2, y = generate_data(equation, size=50)
    model = LinearRegression(fit_intercept=True)

    x = numpy.array(list(zip(x1, x2)))
    x_train, y_train = get_training_data(x, y, percent=60)
    model.fit(x_train, y_train)

    display_results(x1, x2, y, model, generator=equation)

    pyplot.show()


if __name__ == "__main__":
    main()
