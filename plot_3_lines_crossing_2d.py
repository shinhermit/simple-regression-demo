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
from sklearn.linear_model import LinearRegression


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

    def y(self, x):
        if self.b_ != 0:
            return -1/self.b_ * (self.a_*x + self.c_)
        else:
            return 0

    def x(self, y):
        if self.a_ != 0:
            return -1/self.a_ * (self.b_*y + self.c_)
        else:
            return 0

    def reset(self, a=None, b=None, c=None):
        if a is not None:
            self.a_ = a
        if b is not None:
            self.b_ = b
        if a is not None:
            self.c_ = c


def main():
    x = numpy.array([-5, 5])
    f = LineEquation(a=-1, b=1)
    pyplot.plot(x, f.y(x),
                color='blue', linewidth=3)
    pyplot.show()


if __name__ == "__main__":
    main()
