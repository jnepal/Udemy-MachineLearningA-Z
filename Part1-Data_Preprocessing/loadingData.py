# Data Preprocessing

# Importing the library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the data sets
dataset = pd.read_csv('data/Data.csv')

# Matrics of Features

# : means all the lines
# :-1 means all the columns except the last column
X = dataset.iloc[:, :-1].values # Independent Variable
y = dataset.iloc[:, -1:].values # Dependent Variable

# Taking care of missing data
'''
If we look into the data.csv 
the age column data of Spain is Missing
the salary column data of Germany is Missing

How we can handle the missing data
1. We could remove the entire column who's data is missing (not recommended)
2. We could subsitute the data by mean/medain/most_frequent of the entire column data
'''
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3]) # Missing data is in columns 1 and 2.So, fit imputer object in that columns only

# Replacing the missing data
X[:, 1:3] = imputer.transform(X[:, 1:3])

print(X)

# Encoding Categorical data
'''
    Our datasets has two categorical data
    1. Country : Spain, Germany, France
    2. Purchased : Yes, No
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Country let's say X
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) # 0 Column to replace with label encoder

print(X)

''' 
    Since label encoder just encodes the Country by assigning distinct 
    numbers to particular country.
    
    Country, Age, Salary, Purchased
    
    [[0 44.0 72000.0]
     [2 27.0 48000.0]
     [1 30.0 54000.0]
     [2 38.0 61000.0]
     [1 40.0 63777.77777777778]
     [0 35.0 58000.0]
     [2 38.77777777777778 52000.0]
     [0 48.0 79000.0]
     [1 50.0 83000.0]
     [0 37.0 67000.0]]
     
     Number Index:
     0	France
     1	Spain
     2	Germany
     
     But Machine Learning Algorithms might think spain is greater than france
     since it has higher index than France and might result in error.
     
     So, We encode only independent Categorical Data(not numeric) like country as dummy encoding 
     using library OneHotEncoder. No need to dummy encode dependent categorical datas
     like Purchased.
     
     
'''
onehotencoder_X = OneHotEncoder(categorical_features = [0]) # 0 is the index of columns to dummy encode
X = onehotencoder_X.fit_transform(X).toarray(); 

print('After Dummy Encoding Country Column', X)

'''
    We don't need to dummy encode Purchased Column 
    since it is dependent column
'''
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

print('Label Encoded y', y)

# Splitting the dataset into Training set and Test set
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0) # 30% of data is test set

# Feature Scaling
'''
    The numeric data (Age and Salary) are in different scales
    Age is in 10 whereas Salary is in range of K and Some of
    machine learning algorithm uses eculidean distance i.e
    
    distance between p1(x1, y1) and p2(x2, y2) is sqrt((x2-x1)^2 + (y2-y1)^2)
    
    Therefore, the salary will have more effect(more domination) in feature than the age. So, We 
    scaling both the datas in the range [-1:1].
    
    Some of the methods used for Feature Scaling are
    1. Standardisation
    2. Normalisation
    
    Some of the Machine Learning algorithm isn't based on Eculidean distance like decision tree.
    But it's good to scale the features since, processing would be much faster.
    
'''
from sklearn.preprocessing import StandardScaler

standardScaler_X = StandardScaler()
# For Training set will fit and transform 
# whereas for test set we will only transform

'''
    No need to scale the dummy variables although scaling dummy variables will
    bring all the features in same range but we will lose the interpretation
    as we mayn't be able to say particular index is for this variable
    
    Since, We won't make any interpretations of this loaded data, we have left dummy variable
    columns to be scaled.
'''
X_train = standardScaler_X.fit_transform(X_train)
X_test = standardScaler_X.transform(X_test)