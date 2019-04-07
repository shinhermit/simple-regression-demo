"""
Plots the article's example of a 2D regression.
(sectoion "Understanding linearity")
"""

import numpy
import matplotlib.pyplot as pyplot
from sklearn import linear_model

# Genrating random linear data
# There will be 50 data points ranging from 0 to 50
x = numpy.linspace(0, 50, 50)
y = numpy.linspace(0, 50, 50)

# Adding noise to the random linear data
# Seed numpy so as to generate predictable random numbers
numpy.random.seed(101)
x += numpy.random.uniform(-4, 4, 50)
y += numpy.random.uniform(-4, 4, 50)

partition_size=25

# Split the data into training/testing sets
x_train = x[:-partition_size]
y_train = y[:-partition_size]

x_test = x[partition_size:]
y_test = y[partition_size:]

# Create linear regression object
model = linear_model.LinearRegression()

# Train the model using the training sets
#model.fit(x_train, y_train)
model.fit(numpy.array(x_train).reshape(-1, 1),
          numpy.array(y_train).reshape(-1, 1))

# Make predictions using the testing set
y_pred = model.predict(numpy.array(x).reshape(-1, 1))

# Plot outputs
pyplot.scatter(x, y,  color='black')
pyplot.plot(x, y_pred, color='blue', linewidth=3)

pyplot.show()
