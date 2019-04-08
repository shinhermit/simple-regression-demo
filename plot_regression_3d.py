import numpy
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D
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
    numpy.random.seed(seed_x)
    x = numpy.linspace(0, size, size) + numpy.random.uniform(-size * 0.1, size * 0.1, size)
    numpy.random.seed(seed_y)
    y = numpy.linspace(0, size, size) + numpy.random.uniform(-size * 0.1, size * 0.1, size)
    numpy.random.seed(seed_z)
    z = plane_z(x, y) + numpy.random.uniform(-size * 0.2, size * 0.2, size)
    return x, y, z


def get_training_data(x: numpy.ndarray, y: numpy.ndarray, percent=20) -> (numpy.ndarray, numpy.ndarray):
    size = int(x.shape[0] * percent / 100)
    return x[:size], y[:size]


def display_results(x: numpy.ndarray,
                    x1: numpy.ndarray,
                    x2: numpy.ndarray,
                    y: numpy.ndarray,
                    model: callable,
                    generator_equation: callable=None):
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(azim=25, elev=30)
    
    # plot the data
    ax.scatter(x1, x2, y, c='r')

    # for planes
    mesh_dim = numpy.array([x.min(), x.max()])
    mesh_x, mesh_y = numpy.meshgrid(mesh_dim, mesh_dim)

    # plot the trained model
    plane_z = create_plane_equation(ax=model.coef_[0],
                                    by=model.coef_[1],
                                    cz=-1,
                                    dd=model.intercept_)
    ax.plot_surface(mesh_x, mesh_y,
                    plane_z(mesh_x, mesh_y),
                    alpha=0.5,
                    linewidth=0,
                    antialiased=False)

    # plot the data generator equation
    if generator_equation is not None:
        ax.plot_surface(mesh_x, mesh_y,
                        generator_equation(mesh_x, mesh_y),
                        color='y',
                        alpha=0.5,
                        linewidth=0,
                        antialiased=False)


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
