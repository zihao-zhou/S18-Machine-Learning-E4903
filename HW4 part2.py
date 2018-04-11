
# coding: utf-8

# In[90]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Part a

# Read data from csv
train = np.loadtxt('data/ratings.csv', delimiter = ',', dtype = 'double')
test = np.loadtxt('data/ratings_test.csv', delimiter = ',', dtype = 'double')
#train.shape



# In[138]:



M = np.zeros((943, 1682))

# Fill the matrix
for i in range(95000):
    M[int(train[i][0] - 1)][int(train[i][1] - 1)] = train[i][2]
    


# In[117]:


# Probability Matrix Factorization function
def PMF(M, d, sigma2, t):
    size1 = len(M)
    size2 = len(M[0])
    u = np.zeros((size1, d))
    v = np.random.normal(0, 1, (size2, d))
    L = []
    for i in range(t):
        #print(i)
        for j in range(size1):
            ind = np.where([M[j] != 0])[1]
            temp1 = v[ind].T.dot(v[ind])
            temp2 = M[j][ind].reshape(-1, 1) * v[ind]
            u[j] = np.linalg.inv(sigma2 * np.eye(d) + temp1).dot(np.sum(temp2, axis = 0))
        for k in range(size2):
            ind2 = np.where([M.T[k] != 0])[1]
            temp3 = u[ind2].T.dot(u[ind2])
            temp4 = M.T[k][ind2].reshape(-1, 1) * u[ind2]
            v[k] = np.linalg.inv(sigma2 * np.eye(d) + temp3).dot(np.sum(temp4, axis = 0))
        # Likelihood and objective function
        
        X = (M - u.dot(v.T)) * (M != 0)
        l = - np.trace(X.dot(X.T)) / (2 * sigma2) - np.trace(u.dot(u.T)) / 2 - np.trace(v.dot(v.T)) / 2
        L.append(l)
    return u, v, np.array(L)

#res1 = PMF(M, 10, 0.25, 100)

#plt.plot(range(100), res1[2])


# In[116]:


# Run for 10 times
L = []
U = []
V = []
for i in range(10):
    # indicate the time of running
    print(i)
    res = PMF(M, 10, 0.25, 100)
    L.append(res[2])
    U.append(res[0])
    V.append(res[1])
L = np.array(L)
U = np.array(U)
V = np.array(V)


# In[137]:


# Plot
for i in range(10):
    plt.plot(range(99), L[i][1:], label = "{} time figure".format(i), lw = 1)
plt.legend()
plt.grid("True")
plt.xlabel("Iteration")
plt.ylabel("Joint likelihood value")
plt.title("PMF objective function evolution")
plt.savefig("P2_a_new", dpi = 600)


# In[136]:


# Table
RMSE_list = []
for s in range(10):
    RMSE = 0
    T = U[s].dot(V[s].T)
    for i in range(len(test)):
        RMSE += np.square(T[int(test[i][0]) - 1][int(test[i][1]) - 1] - test[i][2])
    RMSE_list.append(np.sqrt(RMSE / 5000))

table = np.zeros((10,2))
for i in range(10):
    table[i][0] = L[i][99]
    table[i][1] = RMSE_list[i]
table = pd.DataFrame(table)
table.columns = ['Train Function', 'Test RMSE']
table = table.sort_values(by = 'Train Function', ascending = False)
table


# In[295]:


# Part b
import heapq

file = open("data/movies.txt","rb")

movies = []
for line in file:
    line = line.decode()[: -1]
    movies.append(line)
movies = np.array(movies)

# "Star Wars" has index 49, "My Fair Lady" 484 and "Goodfellas" 181
def nearest(n, v, k):
    
    diff = v - v[k]
    diff2 = np.square(diff)
    dist2 = diff2.sum(axis = 1)
    ind = np.array(heapq.nsmallest(n + 1, range(len(dist2)), dist2.take)[1:])
    return ind, np.sqrt(dist2[ind])

# here I choose the second training model according to the objective function value
sol1 = nearest(10, V[1], 49)
sol2 = nearest(10, V[1], 484)
sol3 = nearest(10, V[1], 181)

def neighbor(sol, name):
    P = []
    P.append(list(movies[sol[0]]))
    P.append(list(sol[1]))
    df1 = pd.DataFrame(P).T
    df1.columns = ["{} Neighbors".format(name), "Distances"]
    return df1

table1 = neighbor(sol1, "Star Wars -")
table2 = neighbor(sol2, "My Fair Lady -")
table3 = neighbor(sol3, "Goodfellas -")

result = pd.concat([table1, table2, table3], axis=1)
result

