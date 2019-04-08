import numpy
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn import linear_model


def create_plane_equation(ax: float, by: float, cz: float, dd: float) -> callable:
    """
    Create a function to calculate the z coordinate of
    points of a plane.

    :param ax: coefficient of the x coordinate
    :param by: coefficient of the y coordinate
    :param cz: coefficient of the z coordinate
    :param dd: intercept
    :return:
    """
    def plane_z(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
        """
        a*x + b*y + c*z + d = 0
        z = -1/c * (a*x + b*y + d)

        :param x: x coordinate as numpy ndarray
        :param y: y coordinate as numpy ndarray
        :return: z coordinate = -1/c * (a*x + b*y + d)
        """
        a, b, c, d = ax, by, cz, dd
        c_inv = -1/c if c != 0 else 0
        return c_inv * (a*x + b*y + d)

    return plane_z


def generate_data(plane_z: callable, size=50, seed_x=123, seed_y=321, seed_z=1885):
    """
    Generate data points lying close to a plane.

    :param plane_z: function that calculates the z coordinate of the
    points of a plane given their x and y coordinates.
    :param size: the desired size of the generate data set.
    :param seed_x: seed for random noise to add to the x coordinates of the generated data
    :param seed_y: seed for random noise to add to the y coordinates of the generated data
    :param seed_z: seed for random noise to add to the z coordinates of the generated data
    :return: x, y, z values as numpy.ndarray
    """
    numpy.random.seed(seed_x)
    x = numpy.linspace(0, size, size) + numpy.random.uniform(-size * 0.1, size * 0.1, size)
    numpy.random.seed(seed_y)
    y = numpy.linspace(0, size, size) + numpy.random.uniform(-size * 0.1, size * 0.1, size)
    numpy.random.seed(seed_z)
    z = plane_z(x, y) + numpy.random.uniform(-size * 0.2, size * 0.2, size)
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


def display_results(x: numpy.ndarray,
                    x1: numpy.ndarray,
                    x2: numpy.ndarray,
                    y: numpy.ndarray,
                    model: callable,
                    generator_equation: callable=None):
    """
    Plot the results.

    - Plot the data set as a scatter plot
    - Plot the trained model as a plane. We use the coefficients
      a calculated by the model to plot the plane.
    - Plot the generator equation as a plane.

    :param x:the features or inputs of the data set,. x has the shape:
    [[x11, x12],
     [x21, x22]
     ...
     [xn1, xn2]]
    :param x1: the first feature, as a 1D ndarray
    :param x2: the second feature, as a 1D ndarray
    :param y: the target value, as a 1D ndarray
    :param model: the trained scikit learn model
    :param generator_equation: the plane equation used to generate the
    data set.
    """

    def legend_workaround(poly3d: Poly3DCollection):
        """
        Workaround the bug on 3D legend causing the exception:
        > AttributeError: 'Poly3DCollection' object has no attribute '_edgecolors2d'

        This function should be removed when the issue is resolved.

        more:
        - https://stackoverflow.com/a/54994985
        - https://github.com/matplotlib/matplotlib/issues/4067
        """
        poly3d._facecolors2d=poly3d._facecolors3d
        poly3d._edgecolors2d=poly3d._edgecolors3d

    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(azim=25, elev=30)
    
    # plot the data
    ax.scatter(x1, x2, y, c='r', label="Data")

    # for planes
    mesh_dim = numpy.array([x.min(), x.max()])
    mesh_x, mesh_y = numpy.meshgrid(mesh_dim, mesh_dim)

    # plot the trained model
    plane_z = create_plane_equation(ax=model.coef_[0],
                                    by=model.coef_[1],
                                    cz=-1,
                                    dd=model.intercept_)
    surf = ax.plot_surface(mesh_x, mesh_y,
                    plane_z(mesh_x, mesh_y),
                    alpha=0.5,
                    linewidth=0,
                    antialiased=False,
                    label="Trained model")
    legend_workaround(surf)

    # plot the data generator equation
    if generator_equation is not None:
        surf = ax.plot_surface(mesh_x, mesh_y,
                        generator_equation(mesh_x, mesh_y),
                        color='y',
                        alpha=0.5,
                        linewidth=0,
                        antialiased=False,
                        label="Data generator's equation")
        legend_workaround(surf)

    ax.legend()


def main():
    plane_z = create_plane_equation(ax=10, by=2, cz=-10, dd=5)
    x1, x2, y = generate_data(plane_z, size=50)
    x = numpy.array(list(zip(x1, x2)))
    x_train, y_train = get_training_data(x, y)

    model = linear_model.LinearRegression(fit_intercept=True)
    model.fit(x_train, y_train)

    display_results(x, x1, x2, y, model, plane_z)

    pyplot.show()


if __name__ == "__main__":
    main()
