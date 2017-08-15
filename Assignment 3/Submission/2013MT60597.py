
# coding: utf-8

# In[1]:

# Do the imports, so that we write less code
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from time import time
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from PIL import Image
from scipy.stats import multivariate_normal as mvn


# In[2]:

# Prepare and load the data for analysis
mat = scipy.io.loadmat('2013MT60597.mat')
raw_data = mat['data_image']
target = mat['data_labels']
true_labels = target.flatten()
n_samples, n_features = raw_data.shape
n_digits = len(np.unique(target))
k = n_digits
# print(n_digits)


# In[3]:

# Compress the data using PCA
pca = PCA(n_components=20)
pca.fit(raw_data)
transformed_data = pca.transform(raw_data)
transformed_data = scale(transformed_data)
n_samples, n_features = transformed_data.shape


# In[4]:

#Run Kmeans to initialize the centers
model = KMeans(init='k-means++', n_clusters=n_digits, n_init=10);
model.fit_predict(transformed_data)



# In[5]:

# Global variables, which the E-step and M_step will modify 

w_s = np.zeros((k,n_samples)) 
mu_s = np.zeros((k, n_features))
sigma_s = np.zeros((k,n_features,n_features))
phi_s = np.zeros(k)


# In[6]:

def Kmeans_init(): 
    # return mu_s, sigma_s and phi_s
    mu_s = model.cluster_centers_
    phi_s = np.random.rand(k)
    phi_s /= phi_s.sum()
#     sigma_s = np.random.rand(k,n_features,n_features)*10
    for i in range(k):
        sigma_s[i] = np.eye(n_features)
    return mu_s,sigma_s,phi_s


def runGMM():
    DEBUG = False
    likelihood_old = 0.0
    likelihood_new = 0.0    
    likelihood_array = []
    
    # Global variables, which the E-step and M_step will modify 

    w_s = np.zeros((k,n_samples)) 
    mu_s = np.zeros((k, n_features))
    sigma_s = np.zeros((k,n_features,n_features))
    phi_s = np.zeros(k)
    
    #initialize mu_s, sigma_s and phi_s
    # keep mu_s obtained from the kmeans method
    mu_s = model.cluster_centers_
    phi_s = np.random.rand(k)
    phi_s /= phi_s.sum()
    for i in range(k):
        sigma_s[i] = np.eye(n_features)
#################################
#     #Some Testing
#     for j in range(10):
#         mvn_i = mvn(mu_s[j],sigma_s[j])
#         print(mvn_i.logpdf(transformed_data[9]).T)
# #         w_s[i][j] = mvn_i.pdf(transformed_data[j])*phi_s[i]

################################

#     run the loop at max 300 times
    for loop in range(100): 
        print("Running iteration " )
        print(loop+1)
        # E step

        for i in range(k):
            mvn_i = mvn(mu_s[i],sigma_s[i])
            for j in range(n_samples):
                w_s[i][j] = mvn_i.pdf(transformed_data[j])*phi_s[i]
        sums = w_s.sum(0)
        for i in range(n_samples):
            w_s[:,i] /= sums[i]
        if DEBUG:
            print("Printing W_s")
            print(w_s.sum(0))
        


        # M step
        #update the phi_s
        for i in range(k):
            for j in range(n_samples):
                phi_s[i] += w_s[i][j]
        phi_s /= n_samples

        #update the mu_s
        mu_s = np.dot(w_s,transformed_data)
        sums = w_s.sum(1)
        for i in range(k):
            mu_s[i] /= sums[i]
        if DEBUG:
            print("Printing Mu_s")
            print(mu_s)
        
    
        #update the sigma_s
        for j in range(k):
            for i in range(n_samples):
                y = (transformed_data[i] - mu_s[j]).reshape(n_features,1)
                sigma_s[j] += np.dot(y,y.T) * w_s[j][i]
            sigma_s[j] /= w_s[j].sum()
            #Reassign if determinant = 0
            if np.linalg.det(sigma_s[j]) <= 1e-15:
                print("reassigned on iteration ",loop,"matrix",j)
                sigma_s[j] = np.eye(n_features)
                
        if DEBUG:
            print("Printing Sigmas_s")
            print(sigma_s)
        

        # update the LogLikelihood every step        

        likelihood_new = 0.0
        for i in range(n_samples):
            temp = 0
            for j in range(k):
                temp +=  mvn(mu_s[j], sigma_s[j]).pdf(transformed_data[i]) * phi_s[j] 
#                     print(i,j,mvn(mu_s[j], sigma_s[j]).pdf(transformed_data[i]) * phi_s[j] )
            likelihood_new += np.log(temp)
        likelihood_array.append(likelihood_old)
        print(likelihood_new)


        if np.abs(likelihood_old -likelihood_new) < 1:
            print("likelihood reached a minima")
            break
        likelihood_old = likelihood_new
    return w_s, likelihood_array



# In[ ]:


result_w_s , likelihood_array = runGMM()


# In[17]:

def getPredictions(result_w_s):
   
    # take argmax of columns of result_w_s matrix and take transpose of it so that we get a simple array

    result = np.argmax(result_w_s, axis=0)
#     result = np.flatten(result)
    # build matrix whose ith row corresponds to ith cluster_index
    # matrix[i][j] denotes no. of samples of true_label j assigned to cluster i
    matrix = np.zeros((10,10))
    for i in range(n_samples):
        matrix[result[i]][true_labels[i]] += 1
    # take argmax of each row to assign the label to that cluster
    cluster_label = np.zeros(10)
    for i in range(k):
        cluster_label[i] = np.argmax(matrix[i,:])
    # after assigning the cluster labels, get the label predicted by the model using the cluster index assigned      
    predicted = np.zeros(n_samples)
    for i in range(1,n_samples):
        predicted[i] = cluster_label[result[i]]
    return predicted, cluster_label


# get the no. of correctly classified samples
def getAccuray(true_labels,predicted_labels,samples):
    accuracy = 0.0;
    for i in range(1,samples):
        if(true_labels[i] == predicted_labels[i]):
            accuracy +=1
    return accuracy/samples
def getImage(x):
    img =  Image.fromarray(255 - (x.reshape(28, 28)).astype('uint8'))
    plt.imshow(img,cmap='Greys_r')
    plt.show()
    return img


# In[ ]:


predictions,labels = getPredictions(result_w_s)
print(getAccuray(true_labels,predictions,n_samples))




# In[ ]:



