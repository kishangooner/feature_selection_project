# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd
import numpy as np
data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here

def rf_rfe(data):
    x=data.iloc[:,:-1]
    y=data.iloc[:,-1]
    rd=RandomForestClassifier()
    rfe = RFE(rd,17)
    rfe = rfe.fit(x, y)
    columns = x.columns.values
    support = rfe.support_
    columns_with_support = columns[support]
    
   
        
    return(list(columns_with_support))
   



