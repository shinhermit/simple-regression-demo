import numpy
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model


# generate data
numpy.random.seed(101)
a, b, c, d = -1, 1, 2, -4
x1 = numpy.linspace(0, 50, 50) + numpy.random.uniform(-4, 4, 50)
numpy.random.seed(1983)
x2 = numpy.linspace(0, 50, 50) + numpy.random.uniform(-4, 4, 50)
y = -1/c * (a*x1 + b*x2 + d)

# transform inputs into 2D array
x = numpy.array(list(zip(x1, x2)))

# create a training set
x_train = x[:-25]
y_train = numpy.array(y[:-25]).reshape(-1, 1)

# train linear regression object
model = linear_model.LinearRegression()
model.fit(x_train, y_train)

# make predictions
z_pred = model.predict(x)

# plot the solution
fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')

ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# dataset
ax.scatter(x1, x2, y, c='r')

# prediction
surf = ax.plot_surface(x1, x2, z_pred,
                       alpha=0.5,
                       linewidth=0,
                       antialiased=False)

pyplot.show()
