import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('50Startups.csv')

X = dataset.iloc[:,:-1]
y = dataset.iloc[:,4]


#encoding categorical variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
columntransformer = ColumnTransformer([('encoder',OneHotEncoder(),[3])],remainder='passthrough')
X = columntransformer.fit_transform(X)
#avoiding dummy variable
X = X[:,1:]

# splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

# linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)


#optomal model using backward elimination
import statsmodels.api as sm
import statsmodels.iolib.table as smtb
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
t = regressor_OLS.summary().tables[1][2][4]
z = regressor_OLS.summary().tables[1]
print(regressor_OLS.summary().tables[1])

####

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
t = regressor_OLS.summary().tables[1][2][4]
z = regressor_OLS.summary().tables[1]
print(regressor_OLS.summary().tables[1])

####

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
t = regressor_OLS.summary().tables[1][2][4]
z = regressor_OLS.summary().tables[1]
print(regressor_OLS.summary().tables[1])

####

####

X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
t = regressor_OLS.summary().tables[1][2][4]
z = regressor_OLS.summary().tables[1]
print(regressor_OLS.summary().tables[1])

####

X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
t = regressor_OLS.summary().tables[1][2][4]
z = regressor_OLS.summary().tables[1]
print(regressor_OLS.summary().tables[1])
print('###')
# print(regressor_OLS.t_test([0,2]))
print(regressor_OLS.predict(X_test))
#### 