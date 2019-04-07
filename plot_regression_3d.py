import numpy
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model


def display_results(x1, x2, y, y_pred):
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # dataset
    ax.scatter(x1, x2, y, c='r')
    
    # prediction
    #ax.plot_trisurf(x1, x2, y_pred.reshape(-1, 50),
    #                       alpha=0.5,
    #                       linewidth=0,
    #                       antialiased=False)
    X, Y = numpy.meshgrid(x1, x2)
    a, b, c, d = -1, 1, 2, -4
    Z = -1/c * (a*X + b*Y + d) + numpy.random.uniform(-10, 10, 50)
    surf = ax.plot_surface(X, Y, Z,
                           alpha=0.5,
                           linewidth=0,
                           antialiased=False)
    
    pyplot.show()


# generate data
numpy.random.seed(101)
a, b, c, d = -1, 1, 2, -4
x1 = numpy.linspace(0, 50, 50) + numpy.random.uniform(-4, 4, 50)
numpy.random.seed(1983)
x2 = numpy.linspace(0, 50, 50) + numpy.random.uniform(-4, 4, 50)
numpy.random.seed(1885)
y = -1/c * (a*x1 + b*x2 + d) + numpy.random.uniform(-10, 10, 50)

# transform inputs into 2D array
x = numpy.array(list(zip(x1, x2)))

# create a training set
x_train = x[:-25]
y_train = y[:-25]

# train linear regression object
model = linear_model.LinearRegression()
model.fit(x_train, y_train)

# make predictions
y_pred = model.predict(x)

display_results(x1, x2, y, y_pred)
