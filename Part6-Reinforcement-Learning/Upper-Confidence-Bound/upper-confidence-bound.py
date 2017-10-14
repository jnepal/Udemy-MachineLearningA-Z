# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv( 'data/Ads_CTR_Optimisation.csv' )

# Implementing UCB
import math

d  = 10 # No. of Ads
N = 10000 # No.of trials

adsSelected = []
numbersOfSelection = [0] * d # Initialise a empty list to counter no.of times the particular ad was selected
sumOfRewards = [0] * d
totalReward = 0

for n in range( 0, N ):
    ad = 0
    maxUpperBound = 0
    for i in range( 0, d ):
        # if the ads is selected atleast once only execute
        # else we will not have knowledge about the rewards 
        if ( numbersOfSelection[i] > 0 ): 
            averageReward = sumOfRewards[i] / numbersOfSelection[i]
            deltaI = math.sqrt( 3/2 * math.log( n+1 ) / numbersOfSelection[i] )
            upperBound = averageReward + deltaI
        
        else:
            # 10^400 so, that if above if condition fails for first 10 iterations in each trial and
            # ad no 0 is selected for trial 0, ad no 1 is selected for trial 1 and so on
            upperBound = 1e400 
            
        if upperBound > maxUpperBound:
            maxUpperBound = upperBound
            ad = i
            
    adsSelected.append( ad )
    numbersOfSelection[ad] = numbersOfSelection[ad] + 1
    reward = dataset.values[n, ad]
    sumOfRewards[ad] = sumOfRewards[ad] + reward
    totalReward = totalReward + reward
    
# Visualising the results
plt.hist( adsSelected )
plt.title( 'Histograms of ads selections' )
plt.ylabel( 'Ads' )
plt.ylabel( 'No.of times each ads was selected' )
plt.show()
            