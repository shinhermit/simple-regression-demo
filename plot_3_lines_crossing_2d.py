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
        x = P[0] + t*u[0]
        y = P[1] + t*u[1]

        :param p: a point that lies on the line
        :param u: directing vector of the line
        """
        assert p.shape == u.shape and p.shape == (2,)
        self._x0, self._y0, self._ux, self._uy = p[0], p[1], u[0], u[1]

    def x(self, t: numpy.ndarray) -> numpy.ndarray:
        return self._x0 + t*self._ux

    def y(self, t: numpy.ndarray) -> numpy.ndarray:
        return self._y0 + t*self._uy


def configure_plot(t: numpy.ndarray):
    pyplot.xlim(t.min(), t.max())
    pyplot.ylim(t.min(), t.max())
    pyplot.xlabel('X')
    pyplot.ylabel('Y')
    pyplot.legend()
    pyplot.grid()


def draw_line(f, t, color='blue', **kwargs):
    x = f.x(t)
    y = f.y(t)
    pyplot.plot(x, y,
                color=color,
                **kwargs)

    y_max = max(abs(y.max()), abs(y.min()))
    y_lim = pyplot.ylim()
    pyplot.ylim(min(-y_max, y_lim[0]),
                max(y_max, y_lim[1]))


def main():
    p = numpy.array([2, 1])

    f = LineEquation(p, u=numpy.array([1, 1]))
    g = LineEquation(p, u=numpy.array([1, -1]))
    l = LineEquation(p+0.01, u=numpy.array([.2, -1]))
    h = LineEquation(p, u=numpy.array([.2, -1]))

    t = numpy.array([-5, 5])

    configure_plot(t)
    draw_line(f, t, color='red')
    draw_line(g, t, color='green')
    draw_line(l, t, color='blue', linewidth=1, linestyle='dashed')
    draw_line(h, t, color='blue')

    pyplot.show()


if __name__ == "__main__":
    main()
