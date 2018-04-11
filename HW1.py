##########################################################
# HW1 Machine Learning ELEN 4903 - Problem 2
# Zihao Zhou
# 2018/02/03
##########################################################

# import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import data and transfer to np.mat
train_x = pd.read_csv("X_train.csv")
train_x.astype(float)
train_x = np.mat(train_x)
train_y = pd.read_csv("y_train.csv")
train_y.astype(float)
train_y = np.mat(train_y)
test_x = pd.read_csv("X_test.csv")
test_x.astype(float)
test_x = np.mat(test_x)
test_y = pd.read_csv("y_test.csv")
test_y.astype(float)
test_y = np.mat(test_y)

# generate unit matrix
I = [1,1,1,1,1,1,1]
I = np.mat(np.diag(I))

# compute list of omega with lambda from 0 to 5000
omega = [[],[],[],[],[],[],[]]
omega = np.mat(omega)
for i in range(0, 5001) :
    omega = np.hstack((omega, (I * i + train_x.T \
	* train_x).I * train_x.T * train_y))
	
# compute df from lambda in 0 to 5000
df = []
for x in range(0, 5001):
    trace = train_x * (train_x.T * train_x + x * I).I \
	* train_x.T
    trace = trace.trace()
    trace = np.array(trace.tolist()).flatten()
    df.append(trace[0])

# plot 1: 7 coefficients as function of df
for i in range(0,7):
    plt.plot(df, np.array(omega[i]).flatten(),label = i + 1)

plt.legend() 
plt.xlabel(r"df($\lambda$)")
plt.ylabel("Coefficients")
plt.grid(True)
plt.title(r"Problem 2.a - Coefficients and df($\lambda$)")
plt.savefig("problem2_a", dpi = 600)
plt.show()

# RMSE list as a function of lambda from 0 to 50
RMSE_list = []
for i in range(0, 51):
    y_predict = test_x * omega[:,i]
    RMSE = np.sqrt((y_predict - test_y).T * \
	(y_predict - test_y) /42)
    RMSE = np.array(RMSE[0]).flatten()[0]
    RMSE_list.append(RMSE)

# RMSE list as a function of lambda from 0 to 500
RMSE_list_1 = []
for i in range(0, 501):
    y_predict = test_x * omega[:,i]
    RMSE = np.sqrt((y_predict - test_y).T * \
	(y_predict - test_y) /42)
    RMSE = np.array(RMSE[0]).flatten()[0]
    RMSE_list_1.append(RMSE)

# plot 2
plt.plot(range(0,51), RMSE_list)
plt.xlabel(r'$\lambda$')
plt.ylabel(r"$RMSE^2$")
plt.grid(True)
plt.title("Problem 2.c - RMSE for Ridge Regression")
plt.savefig("problem2_c",dpi = 600)
plt.show()

# Polynomial Regresion
# generate new x
new_x_2 = train_x
print (new_x_2.shape)
for i in range(0,6):
    new_x_2 = np.hstack((new_x_2, \
	np.multiply(train_x[:,i], train_x[:,i])))
print (new_x_2.shape)

I_2 = (1,1,1,1,1,1,1,1,1,1,1,1,1)
I_2 = np.mat(np.diag(I_2))
omega_new_2 = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
omega_new_2 = np.mat(omega_new_2)
for i in range(0, 501) :
    omega_new_2 = np.hstack((omega_new_2, (I_2 * i \
	+ new_x_2.T * new_x_2).I * new_x_2.T * train_y))
omega_new_2.shape

test_x_2 = np.hstack((test_x,np.multiply(test_x[:,:6], \
test_x[:,:6])))
test_x_3 = np.hstack((test_x_2, \
np.multiply(np.multiply(test_x[:,:6],test_x[:,:6]),test_x[:,:6])))
print (test_x_3.shape)
print (test_x_2.shape)

RMSE_list_2 = []
for i in range(0, 501):
    y_predict = test_x_2 * omega_new_2[:,i]
    RMSE = np.sqrt((y_predict - test_y).T * (y_predict - test_y) /42)
    RMSE = np.array(RMSE[0]).flatten()[0]
    RMSE_list_2.append(RMSE)

new_x_3 = new_x_2
print (new_x_3.shape)
for i in range(0,6):
    new_x_3 = np.hstack((new_x_3, np.multiply(np.multiply( \
	train_x[:,i], train_x[:,i]),train_x[:,i])))
print (new_x_3.shape)

I_3 = np.ones(19)
I_3 = np.mat(np.diag(I_3))

omega_new_3 = []
for i in range(0,19):
    omega_new_3.append([])

omega_new_3 = np.mat(omega_new_3)
for i in range(0, 501) :
    omega_new_3 = np.hstack((omega_new_3, (I_3 * i \
	+ new_x_3.T * new_x_3).I * new_x_3.T * train_y))
omega_new_3.shape

RMSE_list_3 = []
for i in range(0, 501):
    y_predict = test_x_3 * omega_new_3[:,i]
    RMSE = np.sqrt((y_predict - test_y).T * (y_predict - test_y) /42)
    RMSE = np.array(RMSE[0]).flatten()[0]
    RMSE_list_3.append(RMSE)
    
# Plot 3
plt.plot(range(0, 501), RMSE_list_1, label = "1")
plt.plot(range(0, 501), RMSE_list_2, label = "2")
plt.plot(range(0, 501), RMSE_list_3, label = "3")
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$RMSE^2$")
plt.legend()
plt.grid(True)
plt.title("Problem 2.d - RMSE for Polynomial Regression")
plt.savefig("problem2_d", dpi = 600)
plt.show()