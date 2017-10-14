# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv( 'data/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3 ) # quoting = 3 to discard the decoding error caused due to ""

# Cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

cleanedCorpus = []

for i in range( 0, len( dataset ) ):
    review = re.sub( '[^a-zA-z]', ' ', dataset['Review'][i] )
    review = review.lower()
    review = review.split()
    
    porterStemmer = PorterStemmer()
    review = [porterStemmer.stem( word ) for word in review if not word in set( stopwords.words( 'english' ) )]
    review = ' '.join( review )
    
    cleanedCorpus.append( review )
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer

countVectorizer = CountVectorizer( max_features = 1500 )

# Feature Selection
X = countVectorizer.fit_transform( cleanedCorpus ).toarray() # Independent Variable should be in matrix representation
y = dataset.iloc[:, 1].values # Dependent variables must be in vector representation

# Splitting the dataset into the Training set and Test Set
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.20 )

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit( X_train, y_train )

# Predicting the Test set results
y_pred = classifier.predict( X_test )

# Making confusion matrix
from sklearn.metrics import confusion_matrix

confusionMatrix = confusion_matrix( y_test, y_pred )
