# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 09:23:04 2020

@author: MONALIKA P
"""

#importing libraries
import numpy as np
from matplotlib import pyplot
import pandas as pd

#importing the datasets
dataset = pd.read_csv('train.csv')

x=dataset.iloc[:,0:14].values
y=dataset.iloc[:,14:].values

#spliting the datasets into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = (0.25), random_state = 0)

#importing linear regression and applying to datasets
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

#if you try to plot graph you will get errors

import statsmodels.api as sm
#optimal variable X_opt will contain only optimal variables having an impact on the independent variable
x=np.append(arr=np.ones((333,1)).astype(float),values=x,axis=1)
x_opt = x[:,:]
regressor_OLS = sm.OLS(endog = y,exog = x_opt).fit()
regressor_OLS.summary()

#removal of the predictor having the highest p value
x_opt = x[:,[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14]]
regressor_OLS = sm.OLS(endog = y,exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14]]
regressor_OLS = sm.OLS(endog = y,exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0, 1, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14]]
regressor_OLS = sm.OLS(endog = y,exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14]]
regressor_OLS = sm.OLS(endog = y,exog = x_opt).fit()
regressor_OLS.summary()

x = x_opt

#spliting the datasets into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = (0.25), random_state = 0)

#importing linear regression and applying to datasets
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)