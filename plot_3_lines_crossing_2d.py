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

    def __init__(self, a=1, b=1, c=1):
        """
        Initializes an equation of the form.
        a*x + b*y + c = 0

        :param a: coefficient of the x variable
        :param b: coefficient of the y variable
        :param c: intercept
        """
        self.a_, self.b_, self.c_ = a, b, c

    def __call__(self, x: numpy.ndarray) -> numpy.ndarray:
        return self.y(x)

    def y(self, x: numpy.ndarray) -> numpy.ndarray:
        if self.b_ != 0:
            b_inv = -1/self.b_
        else:
            b_inv = 0
        return b_inv * (self.a_ * x + self.c_)

    def x(self, y: numpy.ndarray) -> numpy.ndarray:
        if self.a_ != 0:
            a_inv = -1/self.a_ * (self.b_*y + self.c_)
        else:
            a_inv = 0
        return a_inv * (self.b_*y + self.c_)

    def reset(self,
              a:float=None,
              b:float=None,
              c:float=None):
        if a is not None:
            self.a_ = a
        if b is not None:
            self.b_ = b
        if a is not None:
            self.c_ = c


def configure_plot(x: numpy.ndarray):
    pyplot.xlim(x.min(), x.max())
    pyplot.ylim(x.min(), x.max())
    pyplot.xlabel('X')
    pyplot.ylabel('Y')
    pyplot.legend()
    pyplot.grid()


def draw_line(x, f, color='blue'):
    y = f(x)
    pyplot.plot(x, y,
                color=color,
                linewidth=3)

    y_max = max(abs(y.max()), abs(y.min()))
    y_lim = pyplot.ylim()
    pyplot.ylim(min(-y_max, y_lim[0]),
                max(y_max, y_lim[1]))


def main():
    f = LineEquation(a=-1, b=1, c=1)
    g = LineEquation(a=-1, b=-1, c=3)
    h = LineEquation(a=1, b=0, c=-2)

    x = numpy.array([-5, 5])

    configure_plot(x)
    draw_line(x, f, color='red')
    draw_line(x, g, color='green')
    draw_line(x, h, color='blue')

    pyplot.show()


if __name__ == "__main__":
    main()
