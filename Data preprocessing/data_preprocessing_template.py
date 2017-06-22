# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values        #Taking all the lines and all the columns except the last one
Y = dataset.iloc[:,3].values          #Retrieving the last column

"""from sklearn.preprocessing import Imputer   #Allows to take care of missing data
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,0]=labelEncoder_X.fit_transform(X[:,0])
oneHotEncoder = OneHotEncoder(categorical_features=[0])
X=oneHotEncoder.fit_transform(X).toarray()
labelEncoder_Y = LabelEncoder()
Y=labelEncoder_Y.fit_transform(Y)"""


from sklearn.cross_validation import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y,test_size = 0.2, random_state=0)
"""
from sklearn.preprocessing import StandardScaler
scale_X= StandardScaler()
X_Train=scale_X.fit_transform(X_Train)
X_Test=scale_X.transform(X_Test) 
"""