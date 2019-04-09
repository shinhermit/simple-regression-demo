"""
Plot side by side:
- 3 lines all crossing at 1 point
- 3 lines crossing no more at a single point
  after a slight change in 1 parameter

Used to illustrate the section "Limits of an algebraical solution"
of the first article about linear regression.
"""
import numpy
from matplotlib import pyplot


class LineEquation:
    """
    Represents the equation of a line.

    An equation of a line has the form:
    a*x + b*y + c = 0
    """

    def __init__(self, p: numpy.ndarray, u: numpy.ndarray):
        """
        Initializes a parametric equation of the form:
        xreset = P[0] + t*u[0]
        y = P[1] + t*u[1]

        :param p: a point that lies on the line
        :param u: directing vector of the line
        """
        assert p.shape == u.shape and p.shape == (2,)
        self._x0, self._y0, self._ux, self._uy = p[0], p[1], u[0], u[1]

    def __call__(self, t: numpy.ndarray) -> (numpy.ndarray, numpy.array):
        """
        Calculate the coordinate(s) of 1 or more points
        lying on the line given a value of the parameter t.

        :param t: value of the parameter of the line.
        :return: x, y coordinate(s) of 1 or more points
        lying on the line.
        """
        return self.x(t), self.y(t)

    def x(self, t: numpy.ndarray) -> numpy.ndarray:
        """
        Calculate the x coordinate(s) of 1 or more points
        lying on the line given a value of the parameter t.

        :param t: value of the parameter of the line.
        :return: x coordinate(s) of 1 or more points
        lying on the line.
        """
        return self._x0 + t*self._ux

    def y(self, t: numpy.ndarray) -> numpy.ndarray:
        """
        Calculate the y coordinate(s) of 1 or more points
        lying on the line given a value of the parameter t.

        :param t: value of the parameter of the line.
        :return: y coordinate(s) of 1 or more points
        lying on the line.
        """
        return self._y0 + t*self._uy

    def update(self, p: numpy.ndarray=None,
              u: numpy.ndarray=None) -> 'LineEquation':
        """
        Update either the anchor point p, the directing vector u
        or both.

        :param p: a point lying on the line.
        :param u: the directing vector of the line
        :return: this line equation, updated.
        """
        if p is not None:
            assert p.shape == (2,)
            self._x0, self._y0 = p[0], p[1]
        if u is not None:
            assert u.shape == (2,)
            self._ux, self._uy = u[0], u[1]
        return self


def configure_plot(t: numpy.ndarray):
    """
    Configures the drawing.

    Set the initial reference frame limits, among other settings.

    :param t: the parameter of the line to draw.
    """
    pyplot.xlim(t.min(), t.max())
    pyplot.ylim(t.min(), t.max())
    pyplot.xlabel('X')
    pyplot.ylabel('Y')
    pyplot.legend()
    pyplot.grid()


def reset_plot_limits(x: numpy.ndarray, y: numpy.ndarray):
    """
    Update the limits of the reference frame given
    the x and y coordinates of some points drawn in the figure.

    :param x: x coordinates of the points
    :param y: y coordinates of the points
    """
    x_max = max(abs(x.max()), abs(x.min()))
    x_lim = pyplot.xlim()
    pyplot.xlim(min(-x_max, x_lim[0]),
                max(x_max, x_lim[1]))

    y_max = max(abs(y.max()), abs(y.min()))
    y_lim = pyplot.ylim()
    pyplot.ylim(min(-y_max, y_lim[0]),
                max(y_max, y_lim[1]))



def draw_line(f: LineEquation, t: numpy.ndarray, **kwargs):
    """
    Plot a line on the figure.

    :param f: parametric equation of a line.
    :param t: values of the parameter to use for the plotting.
    :param kwargs: additional parameters of the plotting function.
    """
    x, y = f(t)
    pyplot.plot(x, y, **kwargs)
    reset_plot_limits(x, y)


def main():
    p = numpy.array([2, 1])

    f = LineEquation(p, u=numpy.array([1, 1]))
    g = LineEquation(p, u=numpy.array([1, -1]))
    h = LineEquation(p, u=numpy.array([.2, -1]))

    t = numpy.array([-5, 5])

    configure_plot(t)
    draw_line(f, t, color='red')
    draw_line(g, t, color='green')
    draw_line(h, t, color='blue')
    draw_line(h.update(p=p+0.01), t,
              color='blue', linewidth=1, linestyle='dashed')
    pyplot.show()


if __name__ == "__main__":
    main()
