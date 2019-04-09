import numpy


class LineParametricEquation:
    """
    Represents the equation of a line.

    An equation of a line has the form:
        x = P[0] + t*u[0]
        y = P[1] + t*u[1]
    where t is the parameter, P is a point lying
    on the line and u is the directing vector of the line.
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

    def __call__(self, t: numpy.ndarray) -> (numpy.ndarray, numpy.array):
        """
        Calculate the coordinate(s) of 1 or more points
        lying on the line given some values of the parameter t.

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
              u: numpy.ndarray=None) -> 'LineParametricEquation':
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


class PlaneParametricEquation:
    """
    Represents the equation of a plane.

    An equation of a plane has the form:
        x = P[0] + t*u[0] + s*v[0]
        y = P[1] + t*u[1] + s*v[1]
        z = P[2] + t*u[2] + s*v[2]
    where t and s are the parameters, P is a point lying
    on the plane, u and v are 2 non-collinear vectors
    lying on the plane.
    """

    def __init__(self, p: numpy.ndarray, u: numpy.ndarray, v: numpy.ndarray):
        """
        Initializes a parametric equation of the form:
        x = P[0] + t*u[0] + s*v[0]
        y = P[1] + t*u[1] + s*v[1]
        z = P[2] + t*u[2] + s*v[2]

        :param p: a point that lies on the plane
        :param u: vector lying on the plane
        :param v: vector lying on the plane, non-collinear to u.
        """
        assert p.shape == u.shape == u.shape == (3,)
        self._x0, self._y0, self._z0 = p[0], p[1], p[2]
        self._ux, self._uy, self._uz = u[0], u[1], u[2]
        self._vx, self._vy, self._vz = v[0], v[1], v[2]

    def __call__(self, t: numpy.ndarray, s: numpy.ndarray) -> (numpy.ndarray, numpy.array, numpy.array):
        """
        Calculate the coordinate(s) of 1 or more points
        lying on the plane given some values of the parameters t and s.

        :param t: value of the first parameter of the plane.
        :param s: value of the second parameter of the plane.
        :return: x, y, z coordinate(s) of 1 or more points
        lying on the line.
        """
        return self.x(t, s), self.y(t, s), self.z(t, s)

    def x(self, t: numpy.ndarray, s: numpy.ndarray) -> numpy.ndarray:
        """
        Calculate the x coordinate(s) of 1 or more points
        lying on the plane given a value of the parameters t and s.

        :param t: value of the first parameter of the plane.
        :param s: value of the second parameter of the plane.
        :return: x coordinate(s) of 1 or more points
        lying on the line.
        """
        return self._x0 + t*self._ux + s*self._vx

    def y(self, t: numpy.ndarray, s: numpy.ndarray) -> numpy.ndarray:
        """
        Calculate the y coordinate(s) of 1 or more points
        lying on the plane given a value of the parameters t and s.

        :param t: value of the first parameter of the plane.
        :param s: value of the second parameter of the plane.
        :return: y coordinate(s) of 1 or more points
        lying on the line.
        """
        return self._y0 + t*self._uy + s*self._vy

    def z(self, t: numpy.ndarray, s: numpy.ndarray) -> numpy.ndarray:
        """
        Calculate the y coordinate(s) of 1 or more points
        lying on the plane given a value of the parameters t and s.

        :param t: value of the first parameter of the plane.
        :param s: value of the second parameter of the plane.
        :return: y coordinate(s) of 1 or more points
        lying on the line.
        """
        return self._z0 + t*self._uz + s*self._vz


class PlaneEquation:
    def __init__(self, a: float, b: float, c: float, d: float):
        assert a + b + c != 0
        self._a, self._b, self._c, self._d = a, b, c, d

    def __call__(self, x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
        return self.z(x, y)

    def z(self, x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
        if self._c == 0:
            if self._a*x + self._b*y + self._d == 0:
                return x + y
            else:
                raise ValueError("The coefficient of z is null. x and y must comply "
                                 "with {a}x + {b}y + {d} = 0".format(a=self._a, b=self._b, d=self._d))
        else:
            return -1/self._c * (self._a*x + self._b*y + self._d)