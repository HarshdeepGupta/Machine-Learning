
# coding: utf-8

# In[126]:

import csv as csv 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
# %matplotlib outline 

# Open up the csv file in to a Python object
data = pd.read_csv('data.csv', header=0)


# dataFrame.dtypes
# dataFrame.info()
# dataFrame.describe()
# dataFrame.x.mean()

# dataFrame['x'].hist(bins=16, range=(-5,5), alpha = .2)
#P.show()
# dataFrame.head()
# type(dataFrame.x)


use only the first 20 data points in your file.
Solve the curve fitting regression problem using error function minimisation.
use a validation approach to characterise the goodness of fit for polynomials of different order
distinguish overfitting,underfitting, and the best fit
obtain an estimate for the noise variance.
# In[118]:

X_train = data.iloc[:,0]
Y_train = data.iloc[:,1]

for i in range(2,6):
    X_train[i-1] = X_train[0]**i





# In[122]:

from sklearn import linear_model
model = linear_model.RidgeCV(alphas=[0],cv = 10)
result = model.fit(X_train,Y_train.rehape(100,1)) 
# result = model.predict(X_train)

#print(model.score(X_train1,Y_train1))
# x_new = np.arange(-5.1, -3.0, 0.1)
# y_new = model.predict(x_new.reshape(x_new.size[0],1))
plt.plot(X_train,Y_train,'o', x_new, y_new)
plt.show()
# error = 0;
# for i in range(0,20): 
#         error += ((y_new1 - Y_train)**2).sum()

# print((error/20)**0.5)



# In[ ]:

#part 1, no regularization

z = np.polyfit(X_train, Y_train,3)
print(z)
hypothesis = np.poly1d(z)

y_new1 = hypothesis(X_train)
 

x_new = np.arange(-5.1, -3.0, 0.1)
y_new = hypothesis(x_new)

plt.plot(X_train,Y_train,'o', x_new, y_new)
plt.show()

error = 0;
for i in range(0,20): 
        error += ((y_new1 - Y_train)**2).sum()

print((error/20)**0.5)


# In[132]:

#code for univariate regression

X,Y = data.x, data.y
#no. of observations
m = Y.size



def h(theta,x):
    '''theta is a column vector, x is a real no
    returns (theta.T)(X)'''
    result = 0
    for i in range(theta.size):
            result += theta[i]*(x**i)
    return result        
            


def cost(y,x,h,theta,lamda):
    #y is the target data, x is the input data, h is the hypothesis, theta is the parameter vector
    m = y.size
    error = 0;
    predictions = h(theta,X)
    #print(predictions)
    sqErrors = (predictions - y) ** 2
    J =  sqErrors.sum() + lamda*np.square(theta).sum()
    J /= m
    return J**(0.5)

def optimize(w):
    return cost(Y,X,h,w,lamda)


#theta[i] = coeff of X**i
#theta is the initial guess
degree = 5;
theta = np.ones(shape=(1+degree,1))
#set the regularization parameter
lamda =  10

print(cost(Y,X,h,theta,lamda))
theta_new = minimize(optimize,theta)
print(cost(Y,X,h,theta_new.x,lamda))

Y_predicted = h(theta_new.x,X)

plt.plot(X, Y_predicted,color="red",label="Predicted")
plt.plot(X,Y,color="blue",label="Original")
plt.legend(loc='upper left')
plt.grid(True)
plt.show()
print(theta_new.x)




# In[136]:

sigma2 = (((Y_predicted-Y)).sum())/100
print(1/sigma2)


# In[ ]:

costArray = np.zeros(shape=(10,10))
def regress(deg, l):
    
    lamda = l
    theta_init = np.ones(shape=(1+deg,1))
    theta_new = minimize(optimize,theta_init)
    return cost(Y_,X,h,theta_new.x,lamda)

for i in range(1,11):
    for j in range(0,10):
        costArray[i-1,j] = regress(i,j*3)


# In[ ]:

print(costArray)

# from mpl_toolkits.mplot3d import Axes3D
# Xaxis = range(0,10)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(Xaxis, Yaxis, costArray[:,0], c='r', marker='o')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# b = costArray[5,:]
# print(b)

# plt.plot(Xaxis,b)
# plt.show()

np.where(costArray == costArray.min())


# In[ ]:

#code for multivariate regression

def feature_normalize(X):
    '''
    Returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    '''
    mean_r = []
    std_r = []

    X_norm = X

    n_c = X.shape[1]
    for i in range(n_c):
        m = mean(X[:, i])
        s = std(X[:, i])
        mean_r.append(m)
        std_r.append(s)
        X_norm[:, i] = (X_norm[:, i] - m) / s

    return X_norm, mean_r, std_r

