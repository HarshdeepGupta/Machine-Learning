
# coding: utf-8

# In[56]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
get_ipython().magic('matplotlib inline')

# Open up the csv file in to a Python object
data = pd.read_csv('train.csv',header = -1)
X_test_raw = pd.read_csv('test.csv',header = -1)

#data.describe()


# In[112]:

X_raw = data.iloc[:,0:18]
Y = data.iloc[:,18]

#prepare the data, add new features
temp = [X_raw, X_test_raw]

X_new = pd.concat(temp)

#attach the square of the features
#not really sure about this one
for i in range(0,18):
    X_new[18+i] = X_new[i]**2


X = X_new.head(X_raw.shape[0])
X_test = X_new.tail(X_test_raw.shape[0])

print(X_test.shape)


# In[91]:

#function to normalize the normalize the features
def feature_normalize(X):
    '''
    Returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1.
    '''
    mean_r = []
    std_r = []

    X_norm = X

    n_c = X.shape[1]
    for i in range(n_c):
        m = np.mean(X.iloc[:, i])
        s = np.std(X.iloc[:, i])
        mean_r.append(m)
        std_r.append(s)
        X_norm.iloc[:, i] = (X_norm.iloc[:, i] - m) / s

    return X_norm, mean_r, std_r;


# feature_normalize(X);
# X.describe()

#define train and cvd data sets
X_train = X.iloc[0:3000, :]
Y_train = Y.iloc[0:3000]

X_cvd = X.iloc[3000:, :]
Y_cvd = Y.iloc[3000:]

# Y_cvd.describe()

#do some plotting
# t = X.iloc[:,9]**5

# plt.scatter(t,Y)
# plt.show()


# In[ ]:




# In[106]:

# from sklearn import linear_model


# model = linear_model.LinearRegression()


# # Train the model using the training sets
# model.fit(X, Y)

# # The coefficients
# print('Coefficients: \n', model.coef_)
# # The mean square error
# print("Residual sum of squares: %.2f"
#       % np.mean((model.predict(X_cvd) - Y_cvd) ** 2))

# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % model.score(X_cvd,Y_cvd))

# # Plot outputs
# plt.scatter(X_cvd.iloc[:,0], Y_cvd,  color='black')
# plt.plot(X_cvd.iloc[:,0], model.predict(X_cvd), color='blue',
#          linewidth=3)

# plt.xticks(())
# plt.yticks(())

# # plt.show()




# In[19]:


result = model.predict(X_test)
# print(type(result))

np.savetxt('sub.csv', result,newline=',')


  


# In[113]:

ridgeCV = linear_model.RidgeCV(alphas=[0.00000000000001,0.0000001,0.0000000000001],cv = 10,normalize=True)
print(ridgeCV.fit(X,Y)) 
print(ridgeCV.alpha_)


# In[114]:

result = ridgeCV.predict(X_test)
np.savetxt('sub.csv', result,newline=',')


# In[6]:

#define the hypothesis function
def h(theta,x):
    #theta is a column vector, x is a real no
    result = 0
    for i in range(theta.size):
            result += theta[i]*(x**i)
    return result        

