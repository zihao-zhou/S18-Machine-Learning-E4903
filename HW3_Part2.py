
# coding: utf-8

# In[2]:


# Homework3
# Boosting
import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

trainX = np.loadtxt("boosting/X_train.csv", dtype=np.str, delimiter=",")
trainY = np.loadtxt("boosting/y_train.csv", dtype=np.str, delimiter=",")
testX = np.loadtxt("boosting/X_test.csv", dtype=np.str, delimiter=",")
testY = np.loadtxt("boosting/y_test.csv", dtype=np.str, delimiter=",")
trainX = trainX.astype(np.float)
trainY = trainY.astype(np.float)
testX = testX.astype(np.float)
testY = testY.astype(np.float)

size1 = len(trainX)
one1 = np.ones(size1)
trainX = np.c_[trainX, one1]
size2 = len(testX)
one2 = np.ones(size2)
testX = np.c_[testX, one2]


# bootstrap is ok
def bootstrap(trainX, trainY, omega):
    index = random.choice(len(trainX), len(trainX), list(omega.flatten()))
    return trainX[index], trainY[index]



# In[34]:


omega = np.array(1 / len(trainX)).repeat(len(trainX))
e_list = []
alpha_list = []
train_err = []
test_err = []
train = 0
test = 0
n = 1500
for i in range(n):
        print(i)
        # bootstrap function is ok
        temp = bootstrap(trainX, trainY, omega)
        train_X = temp[0]
        train_Y = temp[1]
        # linear regression
        w = np.linalg.inv(train_X.T.dot(trainX)).dot(train_X.T).dot(train_Y)
        res = np.sign(trainX.dot(w))
        
        # in the error, value 1 is the wrong element
        error = 1 - abs((res + trainY) / 2)
        # error is epsilon
        error = error.dot(omega.T)
        while error > 0.5:            
            w = - w
            res = np.sign(trainX.dot(w))
            error = 1 - abs((res + trainY) / 2)
            error = error.dot(omega.T)
        alpha = np.log((1 - error) / error) / 2
        alpha_list.append(alpha)
        e_list.append(error)
        print("error",error, "alpha",alpha)
        omega = omega * np.exp(- alpha * res * train_Y)
        omega = omega / np.sum(omega)
        print("omega",omega.max())
        
        #train error
        train += alpha * np.sign(trainX.dot(w))
        pred_train = np.sign(train)
        pred_train = abs((pred_train + trainY) / 2)
        err_train = len(pred_train[pred_train == 0]) / len(pred_train)
        train_err.append(err_train)
        #print(err_train)
        
        #test error
        test += alpha * np.sign(testX.dot(w))
        pred_test = np.sign(test)
        pred_test = abs((pred_test + testY) / 2)
        err_test = len(pred_test[pred_test == 0]) / len(pred_test)
        test_err.append(err_test)
        #print(err_test)
        
#print(train_err, test_err)


# In[35]:


plt.plot(range(n), train_err)
plt.plot(range(n), test_err)
plt.title("Train and test error for each iteration")
plt.xlabel("Iteration")
plt.ylabel("Error")


# In[33]:


plt.plot(range(n), alpha_list)
plt.xlabel("Iteration")
plt.ylabel("alpha")
plt.title("alpha as a function of t")
plt.show()
plt.plot(range(n), e_list)
plt.xlabel("Iteration")
plt.ylabel("epsilon")
plt.title("epsilon as a function of t")
plt.show()

