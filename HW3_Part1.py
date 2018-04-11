
# coding: utf-8

# In[145]:


# Homework3
# Gaussian process
import pandas as pd
import numpy as np

trainX = np.loadtxt("gaussian_process/X_train.csv", dtype=np.str, delimiter=",")
trainY = np.loadtxt("gaussian_process/y_train.csv", dtype=np.str, delimiter=",")
testX = np.loadtxt("gaussian_process/X_test.csv", dtype=np.str, delimiter=",")
testY = np.loadtxt("gaussian_process/y_test.csv", dtype=np.str, delimiter=",")
trainX = trainX.astype(np.float)
trainY = trainY.astype(np.float)
testX = testX.astype(np.float)
testY = testY.astype(np.float)
testY


# In[176]:


# Question a
def K_b(trainX, b):
    K = []
    size = trainX.shape[0]
    #print(size)
    for i in range(0, size):
        X = np.tile(trainX[i],(size,1))
        X = X - trainX
        K.append(np.square(X).sum(axis = 1))
    #print (K)
    K = np.array(K)
    K = np.exp(-K / b)
    return K


def predict(trainX, sigma2, x, trainY, b):
    K = K_b(trainX, b)
    size = len(K)
    #print(size)
    X = np.tile(x,(size,1)) - trainX
    k = np.square(X).sum(axis = 1)
    #print (k)
    k = np.exp(-k / b)
    return np.dot(k, (np.linalg.inv(sigma2 * np.identity(size) + K))).dot(trainY)

def RMSE(trainX, b, sigma2, testX, trainY, testY):
    size = len(testX)
    #print(size)
    sum = 0
    for i in range(0, size):
        res = predict(trainX, sigma2, testX[i], trainY, b)
        error = np.square(res - testY[i])
        sum += error
    return np.sqrt(sum / size)

def RMSE_Table(trainX, trainY, testX, testY):
    #b = range(5, 16, 2)
    #sigma2 = 0.1 * range(1, 11)
    table = np.zeros((6, 10))
    #print (table)
    for i in range(0, 6):
        b = 5 + i * 2
        print (b)
        for sigma2 in range(1, 11):
            table[i][sigma2 - 1] = RMSE(trainX, b, 0.1 * sigma2, testX, trainY, testY)
    return table


# Question b
table = RMSE_Table(trainX, trainY, testX, testY)
print(table)


# In[178]:


print(pd.DataFrame(table))


# In[179]:


np.min(table)


# In[165]:


#np.min(table)
#predict(x4, 2, x4[0], trainY, 5)
def K_b4(trainX, b):
    K = []
    size = trainX.shape[0]
    #print(size)
    for i in range(0, size):
        #X = np.tile(trainX[i],(size,1))
        X = trainX[i] - trainX
        #print(X)
        K.append(np.square(X))
    #print (K)
    K = np.array(K)
    K = np.exp(-K / b)
    return K


def predict4(trainX, sigma2, x, trainY, b):
    K = K_b4(trainX, b)
    #print(K)
    size = len(K)
    #print(size)
    #X = np.tile(x,(size,1)) - trainX
    X = x - trainX
    k = np.square(X)
    #print (k)
    k = np.exp(-k / b)
    return np.dot(k, (np.linalg.inv(sigma2 * np.identity(size) + K))).dot(trainY)
predict4(x4, 2, x4[0], trainY, 5)


# In[175]:


# Question d
import matplotlib.pyplot as plt

def K_b4(trainX, b):
    K = []
    size = trainX.shape[0]
    #print(size)
    for i in range(0, size):
        #X = np.tile(trainX[i],(size,1))
        X = trainX[i] - trainX
        #print(X)
        K.append(np.square(X))
    #print (K)
    K = np.array(K)
    K = np.exp(-K / b)
    return K


def predict4(trainX, sigma2, x, trainY, b):
    K = K_b4(trainX, b)
    #print(K)
    size = len(K)
    #print(size)
    #X = np.tile(x,(size,1)) - trainX
    X = x - trainX
    k = np.square(X)
    #print (k)
    k = np.exp(-k / b)
    return np.dot(k, (np.linalg.inv(sigma2 * np.identity(size) + K))).dot(trainY)

def plot4(trainX, trainY):
    x4 = trainX[:,3]
    y = []
    for i in range(0, len(trainX)):
        #print (i)
        y.append([x4[i], predict4(x4, 2, x4[i], trainY, 5)])
    y = sorted(y, key = lambda x: x[0])
    y = np.array(y)
    #print(y[0])
    plt.plot(x4, trainY, 'o')
    plt.plot(y[:,0], y[:,1])
    
    plt.grid("True")
    plt.title("Problem 1.d - Prediction on car weight")
    plt.xlabel("Car weight")
    plt.ylabel("Mile per gallon")
    
    plt.savefig("Problem1_d", dpi = 600)
    plt.show()
    return y
res = plot4(trainX, trainY)

