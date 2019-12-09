##PROBLEM STATEMENT

## We have a datset which has two columns, one with the years of experience of employee and what is the salary of that person.
## Suppose you are starting a new company and you need to recruit people and the above dataset are the shortlisted candidates.
## Now we need to find out what is the correlation between salary and years of experience.
## Also create model which will tell what is the best fitting line of this relationship.
## Also if the company doesn't know how to set a salary for new employee use this data to set the salary of the employee.







####Simple Linear regression Model

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
#our datatset contains two columns years of experience and salary
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0) ##10 in test

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

###simple linear regression library will take care of feature scaling therefore we dont need to apply the feature scaling manually.

##Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression 
regressor=  LinearRegression()
regressor.fit(X_train,y_train)   ###here the machine is simple linear regression and learning is the data on which it is learning

###Predicting the Test set results
y_pred=regressor.predict(X_test)

##visualising the training results
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')  ###we are training predicted sets of x train that is real experience
plt.title('salary vs experience(Training set)')
plt.xlabel('Years of experince')
plt.ylabel('Salary')
plt.show()
###we see our line is crossing many observations therefore is nearly a very  good model.

##visualising the test results
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')  ###not changing becuase we check our efficiency of tarined model on test results
plt.title('salary vs experience(Test set)')
plt.xlabel('Years of experince')
plt.ylabel('Salary')
plt.show()
###verdict- it is doing quite well in predicitng these values of the test model.