# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values 

from sklearn.cross_validation import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y,test_size = 1/3, random_state=0)

"""
from sklearn.preprocessing import StandardScaler
scale_X= StandardScaler()
X_Train=scale_X.fit_transform(X_Train)
X_Test=scale_X.transform(X_Test) 

"""
#Testing the Linear regression on the train set
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_Train, Y_Train)

#Testing the results with the test set salaries
y_predic = reg.predict(X_Test) 

#Visualising the training set res

plt.scatter(X_Train, Y_Train, color='red')
plt.plot(X_Train,reg.predict(X_Train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualising the test set res

plt.scatter(X_Test, Y_Test, color='red')
plt.plot(X_Train,reg.predict(X_Train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()