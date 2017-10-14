# Polynomial Regression

# Importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/Position_Salaries.csv')

# Feature Engineering
# The independent feature X should be always matrix
X = dataset.iloc[:, 1:2].values
# The dependent feature y should be always vectore
y = dataset.iloc[:, 2].values

'''
Since we have only 10 observations, we won't
divide it into training set and test set 
'''

# Fitting Linear Regression to  the dataset
from sklearn.linear_model import LinearRegression

linearRegressor = LinearRegression()

linearRegressor.fit(X, y)

'''
 Since, Polynomial Linear Regression is a special type
 of Linear Regression. So, It can be constructed by adding
 polynomial terms to the Linear Regression equation
'''
# Fitting Polynomial Regression to the data set
from sklearn.preprocessing import PolynomialFeatures

polyRegressor = PolynomialFeatures(degree = 4)
X_poly = polyRegressor.fit_transform(X) # Adding polynomial features of degree four to independent Feature X

'''
 Fitting the new transformed independent features(with polynomial features)
 into linear Regressor
'''
linearRegressor2 = LinearRegression()

linearRegressor2.fit(X_poly, y)

# Visualing the Linear Regression results
plt.scatter(X, y, color='red')
plt.plot(X, linearRegressor.predict(X), color='blue')
plt.title('Linear Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color='red')
plt.plot(X, linearRegressor2.predict(polyRegressor.fit_transform(X)), color='blue')
plt.title('Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Smoothing the curve of polynomial Regression results
# np.arrange Return evenly spaced values within a given interval.
X_grid = np.arange(min(X), max(X), 0.1) # step of 0.1
X_grid = X_grid.reshape((len(X_grid)), 1) # Converting it into matrix

plt.scatter(X, y, color='red')
plt.plot(X_grid, linearRegressor2.predict(polyRegressor.fit_transform(X_grid)), color='blue')
plt.title('Polynomial Regression')
plt.xlabel('Postition Level')
plt.ylabel('Salary')
plt.show()
