import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset

dataset = pd.read_csv('SimpleRegressionSalaryData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# splitting the data into traning n testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0 )


# Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting test results
y_pred = regressor.predict(X_test)
# print(y_pred)
# print(X_train)
# print(y_train)

# visualizing training data
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color ='blue')
plt.title('salary vs experience (training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()


# visualizing test data
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color ='blue')
plt.title('salary vs experience (test set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()





