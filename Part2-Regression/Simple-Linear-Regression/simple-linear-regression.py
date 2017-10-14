# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datasets
dataset = pd.read_csv('data/Salary_Data.csv');

# Feature Selection
X = dataset.iloc[:, :-1].values # independent variable
y = dataset.iloc[:, 1].values # dependent variable

# Splitting the dataset into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3)

# Fitting Simple Linear Regression
from sklearn.linear_model import LinearRegression
linearRegressor = LinearRegression()

linearRegressor.fit(X_train, y_train)

# Predicting the test results
y_pred = linearRegressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red') # Scatter Plot
plt.plot(X_train, linearRegressor.predict(X_train), color = 'blue') # Plot Regression line
plt.title('Salary v Experience (Trainig Set)')
plt.xlabel('Years of Experience (Training Set)')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red') # Scatter Plot
plt.plot(X_test, y_pred, color = 'blue') # Plotting Regression line
plt.title('Salary v Experience (Testing Set)')
plt.xlabel('Years of Experience (Testing Set)')
plt.ylabel('Salary')
plt.show()

