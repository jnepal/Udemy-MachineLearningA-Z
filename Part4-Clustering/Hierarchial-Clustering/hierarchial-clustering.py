# Hierarchial Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv( 'data/Mall_Customers.csv' )

# Feature Selection
X = dataset.iloc[:, [3,4]].values

# Using the dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch

dendogram = sch.dendrogram( sch.linkage( X, method = 'ward' ) ) # Here ward tries to minimise the value of within cluster variance
plt.title( 'Dendogram' )
plt.xlabel( 'Customers' )
plt.ylabel( 'Euclidean distances' )
plt.show()

# Fitting hierarchial clustering to the dataset
from sklearn.cluster import AgglomerativeClustering

hierarchialClustering = AgglomerativeClustering( n_clusters = 5, affinity = 'euclidean', linkage = 'ward' ) # optimal no of cluster = 5 from dendogram
y_hc = hierarchialClustering.fit_predict( X )

# Visualising the clusters
plt.scatter( X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1' )
plt.scatter( X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2' )
plt.scatter( X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3' )
plt.scatter( X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4' )
plt.scatter( X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5' )

plt.title( 'Clusters of clients' )
plt.xlabel( 'Annual Income (k$) ' )
plt.ylabel( 'Spending Score (1-100)' )
plt.legend()
plt.show()