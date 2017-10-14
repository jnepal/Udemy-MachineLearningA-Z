# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/50_Startups.csv')

# Feature Selection
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding Categorical Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder_X = LabelEncoder()
X[:, 3] = labelEncoder_X.fit_transform(X[:, 3])

onehotEncoder = OneHotEncoder(categorical_features = [3])
X = onehotEncoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the datasets into Training and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

linearRegressor = LinearRegression()
linearRegressor.fit(X_train, y_train)

# Predicting the Test set Resuls
y_pred = linearRegressor.predict(X_test)

# Building the optimal model using Backward Elimination

''' 
    Since we have our slope equation as
    y = mx + c, where c is constant
    The algorithms like LinearRegression automatically adds some constants
    although we don't have a column for constant in our dataset
    
    But the statsmodels.formula.api doesn't . So, We need to add it manually.
    Adding column of constant with value 1 at the beginning of X features
    
'''
import statsmodels.formula.api as sm
X = np.append(np.ones((50, 1)).astype(int), values = X, axis = 1) # astype(int) is added to take care dataerror which will occur if not converted to int; axis = 1 represents column
X_opt = X[:, [0, 1, 2, 3, 4, 5]] # the second paramter is index(columns) we are going to consider as features, first is selecting all the rows

# Creating regressor to fit
regressorOLS = sm.OLS(endog = y, exog = X_opt).fit() # endog is dependent variable, exog is the features to be considered
regressorOLS.summary() # Prints the p value of features taken. Lower P value signifines more importance. Here we have threshold of 0.05 ~ 5%. Any column who'se value is greater than threshold is removed

# Column 2 has highest p value so, we remove it as features
X_opt = X[:, [0, 1, 3, 4, 5]]
regressorOLS = sm.OLS(endog = y, exog = X_opt).fit()
regressorOLS.summary()

# Removing column 1
X_opt = X[:, [0, 3, 4, 5]]
regressorOLS = sm.OLS(endog = y, exog = X_opt).fit()
regressorOLS.summary()

# Removing Column 4
X_opt = X[:, [0, 3, 5]]
regressorOLS = sm.OLS(endog = y, exog = X_opt).fit()
regressorOLS.summary()

# Removing column 5
X_opt = X[:, [0, 3]]
regressorOLS = sm.OLS(endog = y, exog = X_opt).fit()
regressorOLS.summary()


