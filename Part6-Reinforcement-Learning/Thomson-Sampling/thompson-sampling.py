# Thompson Sampling

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv( 'data/Ads_CTR_Optimisation.csv' )

# Implementing the Thompson Sampling
import random

N = 10000 # Total No. of trials
d = 10 # Total No. of versions of ad

adsSelected = []
numbersOfRewards1 = [0] * d
numbersOfRewards0 = [0] * d
totalReward = 0

for n in range( 0, N ):
    ad = 0
    maxRandom = 0
    for i in range( 0, d ):
        randomBeta = random.betavariate( numbersOfRewards1[i] + 1 , numbersOfRewards0[i] + 1 )
        
        if randomBeta > maxRandom:
            maxRandom = randomBeta
            ad = i
    
    adsSelected.append( ad )
    reward = dataset.values[n, ad]
    
    if reward == 1:
        numbersOfRewards1[ad] = numbersOfRewards1[ad] + 1
    else:
        numbersOfRewards0[ad] = numbersOfRewards0[ad] + 1
    
    totalReward =  totalReward + reward
    
# Visualsing the results
plt.hist( adsSelected )
plt.title( 'Histogram of ads selections' )
plt.xlabel( 'Ads' )
plt.ylabel( 'Number of times each ads was selected' )
plt.show()
 