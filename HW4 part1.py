
# coding: utf-8

# In[1]:


import numpy as np
from numpy.linalg import cholesky  
import matplotlib.pyplot as plt


# In[100]:


# Generating data

np.random.seed(0)
mu1 = np.array([[0, 0]]) 
mu2 = np.array([[3, 0]]) 
mu3 = np.array([[0, 3]]) 
Sigma = np.array([[1, 0], [0, 1]])  
R = cholesky(Sigma)  
w = np.array([0.2, 0.5, 0.3])

def GeneratingData(n, mu1, mu2, mu3, Sigma, R, w):
    s1 = np.dot(np.random.randn(100, 2), R) + mu1
    s2 = np.dot(np.random.randn(250, 2), R) + mu2
    s3 = np.dot(np.random.randn(150, 2), R) + mu3
    
    return np.concatenate((s1,s2,s3), axis=0)

data = GeneratingData(500, mu1, mu2, mu3, Sigma, R, w)
data.shape


# In[129]:


# K-means
def K_Means(data, T, K):
    mu = []
    L = []
    minis = 0
    n = len(data)
    c = np.zeros(n)
    # randomly initialize the centroids
    np.random.seed(10)
    for i in range(K):        
        mu.append([np.random.random(), np.random.random()])
    mu = np.array(mu)
    for t in range(T):
    # Update the c type for each point
        for i in range(n):
            diff = mu - data[i]
            diff2 = diff * diff
            dist2 = np.sum(diff2, axis = 1)
            mini = dist2.min()
            minis += mini
            c[i] = np.where(dist2 == mini)[0][0]
        L.append(minis)
        minis = 0
        
    # Update the mu centroids
        for i in range(K):
            mu[i] = np.mean(data[c == i], axis = 0)
            
    return c, mu, L


# In[130]:


# Plot for part a
for i in range(2,6):
    #print(i)
    #sol.append(K_Means(data, 20, i)[2])
    plt.plot(range(1,21), K_Means(data, 20, i)[2], label = i)
#plt.plot(range(1,21), sol)
plt.grid("True")
plt.xlabel("Iteration times")
plt.ylabel("Objective function")
plt.title("K-means objective function evaluation with different K")
plt.legend()
plt.savefig("P1_a", dpi = 600)


# In[164]:


# Plots for part b
res3 = K_Means(data, 20, 3)

plt.scatter(data[:,0], data[:,1], c = res3[0], s = 10, label = "Points")
plt.scatter(res3[1][:,0], res3[1][:,1], c = 'r', marker = 'x', s = 50, label = "Cluster means")
plt.title("K-means with K = 3")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig("P1_b1", dpi = 600)


# In[165]:


res5 = K_Means(data, 20, 5)

plt.scatter(data[:,0], data[:,1], c = res5[0], s = 10, label = "Points")
plt.scatter(res5[1][:,0], res5[1][:,1], c = 'r', marker = 'x', s = 50, label = "Cluster means")
plt.title("K-means with K = 5")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig("P1_b2", dpi = 600)

