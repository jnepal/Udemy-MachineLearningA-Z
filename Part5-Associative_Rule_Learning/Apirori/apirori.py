# Associative Rule Learning ( Apirori )

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv( 'data/Market_Basket_Optimisation.csv', header = None ); # header = None to specify no titles present in csv file

# Preparing the data
# Apirori expects list

transactions = []
for i in range( 0, 7501 ):
    transactions.append( [str( dataset.values[i,j] ) for j in range( 0, 20 )] )

# Training APriori on the dataset
from apyori import apriori

'''
    Here we have dataset of product being bought weekly. So,
    min_support = Product that is being bought n(say 3) daily * 7 (days a week as the dataset is weekly) / Total No. of transaction (7500 in our case)
    
    min_confidence = 0.2 ~ The no.of times the rules should be correct. 0.2 means 20% of time.
    The value for min_confidence should be set not high. Suppose, If we set the value high ( 0.8 ~80% ),
    If in some french stores , mineral water and eggs are the things being bought by every people during summer. 
    Mineral Water because it 's summer and eggs because french people love eggs. since, we have set high min_confidence
    value, our rules show there is relation between eggs and mineral water i.e people buying mineral water will end up
    buying egss as other rules it deducted was not true for 80 % case. 
    
    min_lift : shows relevance of rules
    min_length: Rules composed of atleast 2 products
'''
rules = apriori( transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2 )

# Result
result = list(rules)