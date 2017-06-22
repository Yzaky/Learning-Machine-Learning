import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Startups.csv")

X = dataset.iloc[:,:-1].values        #our IV
Y = dataset.iloc[:,4].values          #DV

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,3]=labelEncoder_X.fit_transform(X[:,3])
oneHotEncoder = OneHotEncoder(categorical_features=[3])
X=oneHotEncoder.fit_transform(X).toarray()

# Avoiding Dummy Var Trap taking one dummy variable away

X=X[:,1:] #removing the first colomn of X

from sklearn.cross_validation import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y,test_size = 0.2, random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_Train,Y_Train)

#Predicting the Test
pred_y=reg.predict(X_Test)

#Backward Elimination
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_optimal=X[:,[0,1,2,3,4,5]]
reg_OLS = sm.OLS(endog=Y,exog=X_optimal).fit()
reg_OLS.summary()
X_optimal=X[:,[0,1,3,4,5]]
reg_OLS = sm.OLS(endog=Y,exog=X_optimal).fit()
reg_OLS.summary()
X_optimal=X[:,[0,3,4,5]]
reg_OLS = sm.OLS(endog=Y,exog=X_optimal).fit()
reg_OLS.summary()
X_optimal=X[:,[0,3,5]]
reg_OLS = sm.OLS(endog=Y,exog=X_optimal).fit()
reg_OLS.summary()
X_optimal=X[:,[0,3]]
reg_OLS = sm.OLS(endog=Y,exog=X_optimal).fit()
reg_OLS.summary()
