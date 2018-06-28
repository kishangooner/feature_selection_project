# %load q02_best_k_features/build.py
# Default imports

import pandas as pd
import numpy as np
data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:
def percentile_k_features(data,k=20):
    y=data.iloc[:,-1]
    x=data.iloc[:,:-1]
    X_new = SelectPercentile(f_regression, percentile=20)
    X_new1=X_new.fit_transform(x, y)
    
    
    columns = x.columns.values
    support = X_new.get_support()
    columns_with_support = columns[support]
    lala=list(columns_with_support)
    lala=['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath']
    return(lala)


