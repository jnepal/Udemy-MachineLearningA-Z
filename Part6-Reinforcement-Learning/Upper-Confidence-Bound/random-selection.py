# Random Selection

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv( 'data/Ads_CTR_Optimisation.csv' )

# Implementing the Random Selection
import random

N = 10000 # No. of trials
d = 10 # No. of versions of ads
adsSelected = []
totalReward = 0

for n in range( 0, N ):
    ad = random.randrange( d )
    adsSelected.append( ad )
    reward = dataset.values[n, ad]
    totalReward= totalReward + reward
    

# Visualising the results - Histogram
plt.hist( adsSelected )
plt.title( 'Histogram of ads selection' )
plt.xlabel( 'Ads' )
plt.ylabel( 'Number of times each ad was selected' )
plt.show()
