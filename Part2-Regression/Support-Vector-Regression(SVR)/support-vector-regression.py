# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/Position_Salaries.csv')

# Feature Selection
X = dataset.iloc[:, 1:2].values # independent features as matrix
y = dataset.iloc[:, 2].values # dependent features as vector

'''
    Since, we do have only 10 observations,
    we wont split the data into training and
    testing set.
'''

'''
    Since, SVR don't have feature scaling by default
    We will scale the feature for better fitting
'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()

# Since we are fitting as well as transforming
# we need to create seperate scaler for x and y

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fitting SVR to the dataset
from sklearn.svm import SVR

SVRRegressor = SVR(kernel = 'rbf') # since our data is non-linear, we could choose kernel among poly and gaussian(rbf)

SVRRegressor.fit(X, y)

# Visualing the SVR Results
plt.scatter(X, y, color='red')
plt.plot(X, SVRRegressor.predict(X), color='blue')
plt.title('SVR Regressor')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Curve Smoothing
# np.arrange Return evenly spaced values within a given interval.
X_grid = np.arange(min(X), max(X), 0.1) # step of 0.1
X_grid = X_grid.reshape((len(X_grid)), 1) # matrix representation

plt.scatter(X, y, color='red')
plt.plot(X_grid, SVRRegressor.predict(X_grid), color = 'blue')
plt.title('SVR Curve Smooting')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting for new value 6.5
'''
    Since we have used feature scaling, we need to feature scale
    the value whose y is to be predicted
    
    X should be matrix. so 6.5 should be converted to matrix representation
'''
y_pred = SVRRegressor.predict(sc_X.transform(np.array([[6.5]])))

# We need to inverse transfrom y_pred to find the real value
y_pred = sc_y.inverse_transform(y_pred)



