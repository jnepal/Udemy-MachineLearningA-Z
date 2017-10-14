# Decision Tree Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv( 'data/Social_Network_Ads.csv' )

# Feature Selection
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into Training and Test set
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.25 )

'''
    We only need to scale the features for those
    algorithms which are based on euclidean distance.
    since, Desicion tree is not based on euclidean distance
    we don't need to but since, we are plotting in higher resolution
    we will here.
'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform( X_train )
X_test = sc_X.transform( X_test )

# Fitting classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
'''
    Our motive is to make the entropy of leaf node close to zero (ideal condition is 0)
    and homogenous as possible. 
    More Homogenity == Less entropy from the parent node
'''
classifier = DecisionTreeClassifier( criterion  = 'entropy' )
classifier.fit( X_train, y_train )

# Predicting the Test Set Results
y_pred = classifier.predict( X_test )

# Confusion Matrix
from sklearn.metrics import confusion_matrix

confusionMatrix = confusion_matrix( y_test, y_pred )

# Visaulising the Training set results
from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid( 
                        np.arange( start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01 ),
                        np.arange( start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01 ) 
                    )
                    
plt.contourf( X1, X2, classifier.predict( np.array( [ X1.ravel(), X2.ravel() ] ).T).reshape( X1.shape ),
             alpha = 0.75, cmap = ListedColormap( ( 'red', 'green' ) ) )
plt.xlim( X1.min(), X1.max() )
plt.ylim( X2.min(), X2.max() )

for i, j in enumerate( np.unique( y_set ) ):
    plt.scatter( X_set[ y_set == j, 0 ], X_set[ y_set == j, 1 ], c = ListedColormap( ( 'red', 'green' ) )(i), label = j )
    
plt.title( 'Decision Tree Classifier ( Training Set ) ' )
plt.xlabel( 'Age' )
plt.ylabel( 'Estimated Salary' )
plt.legend()
plt.show()

# Visulising the Test set results
from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid( 
                        np.arange( start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01 ),
                        np.arange( start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01 ) 
                    )
                    
plt.contourf( X1, X2, classifier.predict( np.array( [ X1.ravel(), X2.ravel() ] ).T).reshape( X1.shape ),
             alpha = 0.75, cmap = ListedColormap( ( 'red', 'green' ) ) )
plt.xlim( X1.min(), X1.max() )
plt.ylim( X2.min(), X2.max() )

for i, j in enumerate( np.unique( y_set ) ):
    plt.scatter( X_set[ y_set == j, 0 ], X_set[ y_set == j, 1 ], c = ListedColormap( ( 'red', 'green' ) )(i), label = j )
    
plt.title( 'Decision Tree Classifier ( Test Set ) ' )
plt.xlabel( 'Age' )
plt.ylabel( 'Estimated Salary' )
plt.legend()
plt.show()
