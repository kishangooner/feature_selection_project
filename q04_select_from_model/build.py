# %load q04_select_from_model/build.py
# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')


def select_from_model(data):
    
    x=data.iloc[:,:-1]
    y=data.iloc[:,-1]
    rd=RandomForestClassifier()
    
    rd.fit(x,y)
    model =SelectFromModel(rd,prefit=True)
    
    X_new=model.transform(x)
    columns = x.columns.values
    support = model.get_support()
    columns_with_support = columns[support]
    return(list(columns_with_support))



