import numpy
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model


def train_and_make_prediction(x1, x2, y):
    x = numpy.array(list(zip(x1, x2)))
    #
    x_train = x[:-25]
    y_train = numpy.array(y[:-25]).reshape(-1, 1)
    # Create linear regression object
    model = linear_model.LinearRegression()
    # Train the model using the training sets
    model.fit(x_train, y_train)
    return model.predict(x)


a, b, c, d = -1, 1, 2, -4
x = numpy.linspace(0, 50, 50)
y = numpy.linspace(0, 50, 50)
z = -1/c * (a*x + b*y + d)

numpy.random.seed(101)
feature1 = x + numpy.random.uniform(-4, 4, 50)
feature2 = y + numpy.random.uniform(-4, 4, 50)
target = z + numpy.random.uniform(-4, 4, 50)

# Make predictions using the testing set
z_pred = train_and_make_prediction(feature1, feature2, target)

fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')

ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

surf = ax.plot_surface(x, y, z_pred,
                       alpha=0.5,
                       linewidth=0,
                       antialiased=False)

ax.scatter(feature1, feature2, target, c='r')

pyplot.show()
