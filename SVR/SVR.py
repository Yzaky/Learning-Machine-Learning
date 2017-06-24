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

from sklearn.preprocessing import StandardScaler
scale_X= StandardScaler()
scale_Y= StandardScaler()
X=scale_X.fit_transform(X)
Y=scale_Y.fit_transform(Y) 


#rbc since our problem is not linear
from sklearn.svm import SVR
regressor = SVR(kernel ='rbf' )
regressor.fit(X,Y)

y_pred=scale_Y.inverse_transform(regressor.predict(scale_X.transform(np.array([6.5]))))



plt.scatter(X,Y, color = 'red')
plt.plot(X,regressor.predict(X),color = 'blue')
plt.title('Truth or Bluff ( Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()