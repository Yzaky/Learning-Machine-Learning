# Polynomial linear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values 

#Creating a linear regression
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X,Y)

#Fitting the polynomial regression into the dataset.
from sklearn.preprocessing import PolynomialFeatures
polynomial_regressor = PolynomialFeatures(degree = 2)
X_polynomial = polynomial_regressor.fit_transform(X)

linear_regressor2 = LinearRegression()
linear_regressor2.fit(X_polynomial,Y)

plt.scatter(X,Y, color = 'red')
plt.plot(X,linear_regressor.predict(X),color = 'blue')
plt.title('Truth or Bluff ( Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

plt.scatter(X,Y, color = 'red')
plt.plot(X,linear_regressor2.predict(X_polynomial),color = 'blue')
plt.title('Truth or Bluff ( Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

linear_regressor.predict(6.5)
linear_regressor2.predict(polynomial_regressor.fit_transform(6.5))