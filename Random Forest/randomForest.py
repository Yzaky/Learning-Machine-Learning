# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values 


"""from sklearn.cross_validation import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y,test_size = 1/3, random_state=0)
"""


#regressor decision Tree
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=13, random_state=0)
regressor.fit(X,Y)


y_pred = regressor.predict(6.5)

X_grid = np.arange(min(X), max(X),0.011)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title("Truth or Bluff (random forest regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
