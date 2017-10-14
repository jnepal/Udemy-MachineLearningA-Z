# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/Position_Salaries.csv')

# Feature Selection
X = dataset.iloc[:, 1:2].values # Independent variable better be in matrix representation
y = dataset.iloc[:, 2].values # Dependent variable in vector representation

'''
Since we have only 10 observations, 
we won't divide into testing and training set
'''

# Fitting the Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor

decisionTreeRegressor = DecisionTreeRegressor()
decisionTreeRegressor.fit(X, y)

# Visualising the Decision Tree Regression results

'''
Unlike other regression like Multiple Linear Regression,
Simple Linear Regression, Support Vector Regression etc.
Decision Tree is non continuous.So, High Resolution graph
and graph smoothing is must.
'''
X_grid = np.arange(min(X), max(X), 0.01) # step of 0.01
X_grid = X_grid.reshape((len(X_grid), 1)) # Matrix representation
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, decisionTreeRegressor.predict(X_grid), color = 'blue')
plt.title('Decision Tree Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


