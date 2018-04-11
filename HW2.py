
# coding: utf-8


import numpy as np
import pandas as pd
from scipy.special import expit
from math import *
from matplotlib import pyplot as plt
from matplotlib.pyplot import stem
from prettytable import PrettyTable
import heapq

train_x = np.loadtxt('X_train.csv', delimiter=',', dtype='int')
train_x = pd.DataFrame(train_x)
train_y = np.loadtxt('y_train.csv', delimiter=',', dtype='int')
train_y = pd.DataFrame(train_y)
#train_x = pd.read_csv("X_train.csv")
#train_y = pd.read_csv("y_train.csv")
test_x = np.loadtxt('X_test.csv', delimiter = ',', dtype = 'int')
test_x = pd.DataFrame(test_x)
test_y = np.loadtxt('y_test.csv', delimiter = ',', dtype = 'int')
test_y = pd.DataFrame(test_y)

# test for input

print(train_x.head(5))
#print(train_y.head(5))
print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)



# Some helper functions
def toLog(a):
    #a = toFloat(a)
    b = []
    for i in a:
        b.append(log(i))
    return pd.DataFrame(b)
def toFloat(s):
    for i in range(0, len(s)):
        s[i] = float(s[i])
    return s



# Naive Bayesian Classifier
def train(train_x, train_y):
    #parameters = []
    x_1 = []
    x_0 = []
    for i in range(0, len(train_x)):
        if train_y.iloc[i][0] == 1:
            x_1.append(train_x.iloc[i])
        else:
            x_0.append(train_x.iloc[i])
    #print(len(x_1))
    #print(len(x_0))
    x_1 = pd.DataFrame(x_1)
    x_0 = pd.DataFrame(x_0)
    #print(x_1)
    #print(x_0)
    
    x_1_binary = x_1.iloc[:,:54]
    x_0_binary = x_0.iloc[:,:54]
    x_1_continue = x_1.iloc[:,54:]
    x_0_continue = x_0.iloc[:,54:]
    #print(x_1_continue.head(5))
    #print(x_0_continue.head(5))
    
    sum_1 = x_1_binary.apply(lambda x: x.sum())
    #print(sum_1)
    sum_1 = toFloat(sum_1)
    
    pi_1 = sum_1 / len(x_1)
    sum_0 = x_0_binary.apply(lambda x: x.sum())
    #print(sum_0)
    sum_0 = toFloat(sum_0)
    pi_0 = sum_0 / len(x_0)
    
    #x_1_continue = float(x_1_continue)
    x_1_continue_log_sum = x_1_continue.apply(lambda x: toLog(x).sum())
    x_0_continue_log_sum = x_0_continue.apply(lambda x: toLog(x).sum())
    #lnSum_1 = x_1_continue.apply(lambda x: x.sum())
    
    #print(pi_1.head(5))
    #print(pi_0.head(5))
    #print(x_1_continue_log_sum)
    #print(x_0_continue_log_sum)
    
    continue_1 = len(x_1) / x_1_continue_log_sum
    continue_0 = len(x_0) / x_0_continue_log_sum
    
    #print(continue_0)
    #print(continue_1)
    return [pi_1, pi_0, continue_1, continue_0, len(x_0) / len(train_x)]
    
s = train(train_x, train_y)
print(s[])
#print(s[3])



def test(test_x, test_y):
    table = np.zeros((2,2))
    #print(table[0,0], table[0,1], table[1,0], table[1,1])
    for i in range(0, len(test_x)):
        x = test_x.iloc[i]
        y = test_y.iloc[i]
        #print (x)
        p_1 = 1 - s[4]
        p_0 = s[4]
        for j in range(0, 54):
            p_1 *= pow(s[0][j], x[j]) * pow(1 - s[0][j], 1 - x[j])
            p_0 *= pow(s[1][j], x[j]) * pow(1 - s[1][j], 1 - x[j])
        for j in range(54, 3):
            p_1 *= s[2][j] * pow(x[j], - s[2][j] - 1)
            p_0 *= s[3][j] * pow(x[j], - s[3][j] - 1)
        predict = (p_1 > p_0)
        
        if (predict == True): 
            predict = 1
        else: 
            predict = 0
        #print (predict)
        table[y, predict] += 1
    return table

table = test(test_x, test_y)




pt = PrettyTable(["", "predict_as_non", "predict_as_spam"])
pt.add_row(["non_spam", table[0, 0], table[0, 1]])
pt.add_row(["spam", table[1, 0], table[1, 1]])
print(pt)
print("Precision: ", (table[0, 0] + table[1, 1]) / len(test_x))


#from pylab import *

#p_0 = stem(range(0, 54), s[0], 'g',markerfmt='bo', '-.')
#p_1 = stem(range(0, 54), s[1], 'b', markerfmt='go', ':')
p_0 = stem(range(0, 54), s[0], '-.')

print(p_0)
print(p_1)
plt.setp(markerline, 'markerfacecolor', 'b')
plt.setp(baseline, 'color','r', 'linewidth', 2)

p_1 = stem(range(0, 54), s[1], ':')
plt.setp(markerline, 'markerfacecolor', 'g')
plt.setp(baseline, 'color','g', 'linewidth', 2)



stem(range(0, 54), s[1], 'b', markerfmt='bo', label='non_spam')
stem(range(0, 54), s[0], 'g', markerfmt='go', label='spam')
#plt.setp(baseline, 'color','r', 'linewidth', 2)
#plt.setp(markerline, 'markerfacecolor')
plt.legend()
plt.title("Problem 2.b - Stem plot for 54 bernoulli parameters")
plt.grid("True")
plt.savefig("problem2_b", dpi = 600)
plt.show()

# KNN classifier

def KNN(train_x, train_y, x, k):
    distances = []
    for i in range(0, len(train_x)):
        #print(i)
        #distances.append((L1_distance(train_x.iloc[i], x), train_y.iloc[i][0]))
        distances.append([np.sum(abs(x - train_x.iloc[i])), train_y.iloc[i][0]])
    #print(distances[:k])
    top_k = heapq.nsmallest(k, distances, key=lambda distances: distances[0])
    t = np.sum(pd.DataFrame(top_k))[1]
    #print(len(top_k))
    #top_k = pd.DataFrame(top_k)
    #t = top_k[top_k[1] == 1]
    #print(top_k)
    #print(top_k[top_k[1] == 1])
    #print(t)
    #print(top_k)
    return int(t > k - t)

#print(test_x.shape)
#print(len(train_x), len(train_y))
KNN(train_x, train_y, test_x.iloc[9], 10)

# try and test
def L1_distance(a, b):
    d = 0
    for i in range(0, len(a)):
        #print(i)
        d += abs(a[i] - b[i])
    return d

print(train_y.head(5))
print(train_y.iloc[:5][0])




def KNN_test(train_x, train_y, test_x, test_y, k):
    count = 0
    for i in range(0, len(test_x)):
        judge = int(KNN(train_x, train_y, test_x.iloc[i], k) == test_y.iloc[i][0])
        count += judge
        #print(i, judge)
    #print(count)
    return count / len(test_x)

k = 10
KNN_test(train_x, train_y, test_x, test_y, k)


def KNN_test_k(train_x, train_y, test_x, test_y, start, end):
    sol = []
    for i in range(start, end + 1):
        f = KNN_test(train_x, train_y, test_x, test_y, i)
        sol.append(f)
        print(i, f)
    return sol
sol = KNN_test_k(train_x, train_y, test_x, test_y, 1, 20)



plt.plot(range(1, 21), sol, '--o')
plt.grid("True")
plt.title("Problem 2.c - KNN precision from 1 to 20")
plt.savefig("problem2_c", dpi = 600)
plt.show()


# logistic Regression

# pre-processing
new_y0 = train_y
new_y0[new_y0 == 0] = -1
new_x0 = train_x
new_x0[57] = 1

new_y1 = test_y
new_y1[new_y1 == 0] = -1
new_x1 = test_x
new_x1[57] = 1


omega = np.zeros(58)

def sigma(new_y0, new_x0, i, w):
    t = array(np.mat(new_x0.iloc[i]) * np.mat(w).T)[0][0]
    s = t / (t + 1)
    if new_y0.iloc[i][0] == 1:
        return s
    else:
        return 1 - s
#sigma(new_y0, new_x0, 1, omega)


def diff(new_y0, new_x0, w):
    sol = 0
    for i in range(0, len(new_y0)):
        sol += sigma(new_y0, new_x0, i, w) * new_y0.iloc[i][0] * new_x0.iloc[i]
    return sol

def LinearReg(x, y, w0, re):
    for i in range(0, re):
        w0 += (pow(10, -5) / sqrt(i + 1)) * diff(y, x, w0)
    return w0
LinearReg(new_x0, new_y0, omega, 5)



#x0 = np.mat(new_x0)
def logistic_train_diff(x0, y0, w0, L):
    x = np.mat(x0) * np.mat(w0).T
    sig = expit(np.multiply(x, y0))
    new_sig = sig + 0.00000000000000000001
    mat4 = 1 - sig
    L.append(np.sum(np.log(new_sig)))
    mat3 = np.mat(multiply(y0, mat4))
    temp = mat3
    for i in range(0, 57):
        temp = hstack((temp, mat3))
    mat5 = np.multiply(temp, np.mat(new_x0))
    s = pd.DataFrame(mat5).sum()
    #print(s)
    return s,L

#logistic_train_diff(new_x0, new_y0, w)
def logistic_train(k, x0, y0, w0):
    L = []
    for i in range(0, k):
        w0 += (pow(10, -5) / sqrt(i + 1)) * logistic_train_diff(x0, y0, w0, L)[0]
    return w0, L

# test
w = np.zeros(58)
k = 1
logistic_res = logistic_train(k, new_x0, new_y0, w)
L_value = logistic_res[1]
w_value = logistic_res[0]


#print(L_value)
plt.plot(range(0, k), L_value)
plt.title("Problem 2.d - Logistic Regression with steepest ascent")
plt.xlabel("Iteration times")
plt.ylabel("Objective training function L")
plt.grid("True")
plt.savefig("problem2_d", dpi = 600)
plt.show()

#x_1 = new_x1.iloc[2]
#h = np.mat(new_x0.iloc[2]) * np.mat(logistic_res).T


# Newton's Method
def logistic_train_diff2(x0, w0):
    
    x = np.mat(x0)
    x0 = x    
    w = np.mat(w0)    
    x = x * w.T    
    sig = expit(x) + 0.00000000000000000001
    temp = np.multiply(sig, 1 - sig)
    matrix = np.mat(zeros((58,58)))
    for i in range(0, len(x0)):
        #print(x0[i])
        #print(np.array(temp)[i][0])
        matrix -= np.array(temp)[i][0] * (x0[i].T * x0[i])
    
    #print(matrix)
    return matrix


#logistic_train_diff2(new_x0, w_2)

def logistic_train_newton(w, x, y, k):
    L = []
    w0 = np.mat(w)
    for i in range(0, k):
        print(i)
        #print((logistic_train_diff2(x, w0).I * np.mat(np.array(logistic_train_diff(x, y, w0, L)[0])).T).shape)
        w0 -= (1 / sqrt(i + 1)) * (logistic_train_diff2(x, w0).I * np.mat(logistic_train_diff(x, y, w0, L)[0]).T).T
    return w0, L

# test

w_2 = np.zeros(58)
k_2 = 100
res = logistic_train_newton(w_2, new_x0, new_y0, k_2)
w_val = res[0]
L_val = res[1]



#print(w_val)
plt.plot(range(0, k_2), L_val)
plt.title("Problem 2.e - Logistic Regression with Newton Method")
plt.xlabel("Iteration times")
plt.ylabel("Objective training function L")
plt.grid("True")
plt.savefig("problem2_e", dpi = 600)
plt.show()

def logistic_test(new_x1, new_y1, w_val):
    table = np.zeros((2,2))
    p1 = expit(np.mat(new_x1) * np.mat(w_val).T)
    p1[p1 > 0.5] = 1
    p1[p1 <= 0.5] = 0
    true_pos = 0
    new_y1[new_y1 == -1] = 0
    #print(np.array(p1)[0][0])
    for i in range(0, len(new_x1)):
        #print(new_y1.iloc[0][0])
        table[new_y1.iloc[i][0]][int(p1[i][0])] += 1
    return table, (table[0][0] + table[1][1]) / len(new_x1)

# test
ans = logistic_test(new_x1, new_y1, w_val)
precision = ans[1]
table2 = ans[0]

# create table
pt = PrettyTable(["", "predict_as_non", "predict_as_spam"])
pt.add_row(["non_spam", table2[0, 0], table2[0, 1]])
pt.add_row(["spam", table2[1, 0], table2[1, 1]])
print(pt)
print("Precision: ", precision)
