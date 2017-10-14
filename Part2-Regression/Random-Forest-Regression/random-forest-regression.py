# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datasets
datasets = pd.read_csv( 'data/Position_Salaries.csv' )

# Feature Selection
X = datasets.iloc[:, 1:2].values # Indpendent Variable better be in matrix representation
y = datasets.iloc[:, 2].values # Dependent Variable better be in vector representation

'''
 Since, We have only 10 observations, we won't divide
 the data into training and testing set
'''

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor

randomForestRegressor = RandomForestRegressor( n_estimators = 300 ) # n_estimators estimates no. of trees in forest

randomForestRegressor.fit( X, y )

'''
 Since, random forest is non-continuous unlike Simple Linear Regression,
 Multiple Linear Regression, Support Vector Regression. So Smoothing curve
 and high resolution is must.
'''
# np.arange returns evenly spaced values within given interval
X_grid = np.arange( min( X ), max( X ), 0.01 ) # Step of 0.01
X_grid = X_grid.reshape( (len( X_grid ), 1) ) # converting it into matrix representation

plt.scatter( X, y, color = 'red' )
plt.plot( X_grid, randomForestRegressor.predict( X_grid ), color = 'blue' )
plt.title( 'Random Forest Regression' )
plt.xlabel( 'Position Level' )
plt.ylabel( 'Salary' )
plt.show()

# Predicting for a new result
y_pred = randomForestRegressor.predict(6.5)