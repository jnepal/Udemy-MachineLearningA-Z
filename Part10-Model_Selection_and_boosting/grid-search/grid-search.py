# Finding optimalparameters using Grid Search

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv( 'data/Social_Network_Ads.csv' )

# Feature Selection
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training and Test set
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.25 )

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform( X_train )
X_test = sc_X.transform( X_test )


# Fitting SVM to the training set
from sklearn.svm import SVC

classifier = SVC( kernel = 'rbf' )
classifier.fit( X_train, y_train )

# Predicting 
y_pred = classifier.predict( X_test )

# Making the confusion matrix
from sklearn.metrics import confusion_matrix

confusionMatrix = confusion_matrix( y_test, y_pred )

# Applying k-Fold cross validation
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score( estimator = classifier, X = X_train, y = y_train, cv = 10 )

accuracies.mean() # Mean of Accuracies
accuracies.std() # Standard deviation of accuracies

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV

parameters = [
                {'C': [1, 10, 100], 'kernel': ['linear']},
                {'C': [1, 10, 100], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.01, 0.001, 0.0001]}
              ]

gridSearch = GridSearchCV(
                            estimator = classifier,
                            param_grid = parameters,
                            scoring = 'accuracy',
                            cv = 10,
                            n_jobs = -1
                          )
gridSearch = gridSearch.fit( X_train, y_train )
bestAccuracy = gridSearch.best_score_
bestParameters = gridSearch.best_params_

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
    
plt.title( 'Support Vector Classifier ( Training Set ) ' )
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
    
plt.title( 'Support Vector Classifier ( Test Set ) ' )
plt.xlabel( 'Age' )
plt.ylabel( 'Estimated Salary' )
plt.legend()
plt.show()
