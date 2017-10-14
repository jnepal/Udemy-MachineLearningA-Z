# Principal Component Analysis in Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv( 'data/Wine.csv' )

# Feature Selection
X = dataset.iloc[:, 0:13].values # Independent variables better be in matrix representation
y = dataset.iloc[:, 13].values # Dependent variable better be in vector form

# Splitting the dataset into the Training and Testing Set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2 )
# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform( X_train )
X_test = sc.transform( X_test )

# Applying PCA
from sklearn.decomposition import PCA

'''
finding out the no.of components which best represents the 
most variance based on explained cummalative variance ratio
'''

pca = PCA( n_components = None )
X_train = pca.fit_transform( X_train )
X_test = pca.transform( X_test )

explained_variance = pca.explained_variance_ratio_

'''
Here we select only 2 principal components as it will represent
 more than 56 % of variance
'''

pca = PCA( n_components = 2 )
X_train = pca.fit_transform( X_train )
X_test = pca.transform( X_test )

# Fitting Logistic Regression to the Training Set
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit( X_train, y_train )

# Predicting the Test set results
y_pred = classifier.predict( X_test )

# Making the confusion matrix
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
             alpha = 0.75, cmap = ListedColormap( ( 'red', 'green', 'blue' ) ) )
plt.xlim( X1.min(), X1.max() )
plt.ylim( X2.min(), X2.max() )

for i, j in enumerate( np.unique( y_set ) ):
    plt.scatter( X_set[ y_set == j, 0 ], X_set[ y_set == j, 1 ], c = ListedColormap( ( 'red', 'green', 'blue' ) )(i), label = j )
    
plt.title( 'Logistic Regression Classifier ( Training Set ) ' )
plt.xlabel( 'PC1' )
plt.ylabel( 'PC2' )
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
             alpha = 0.75, cmap = ListedColormap( ( 'red', 'green', 'blue' ) ) )
plt.xlim( X1.min(), X1.max() )
plt.ylim( X2.min(), X2.max() )

for i, j in enumerate( np.unique( y_set ) ):
    plt.scatter( X_set[ y_set == j, 0 ], X_set[ y_set == j, 1 ], c = ListedColormap( ( 'red', 'green', 'blue' ) )(i), label = j )
    
plt.title( 'Logistic Regression Classifier ( Training Set ) ' )
plt.xlabel( 'PC1' )
plt.ylabel( 'PC2' )
plt.legend()
plt.show()

