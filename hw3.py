#!/usr/bin/env python
# coding: utf-8

# In[116]:


# You are to implement a 3-nearest-neighbor learner for classification.
# To simplify your work, your program can assume that:
# - each item has $d$ continuous features $\x \in \RR^d$
# - binary classification and the class label is encoded as $y \in \{0,1\}$
# - data files are in plaintext with one labeled item per line, separated by whitespace:
#     x_11 ... x_1d y_1
#     x_n1 ... x_nd y_n

# Your program should implement a 3NN classifier: 
# - Use Mahalanobis distance d_A parametrized by a positive semidefinite (PSD) diagonal matrix A.
# - If multiple training points are the equidistant nearest neighbors of a test point,
#   you may use any three of those training points to predict the label.
# - You do not have to implement kd-tree.

import numpy as np
import matplotlib.pyplot as plt
import random


# In[27]:


def distance(x1,x2):
    return np.linalg.norm(x1-x2)


# In[148]:


dataA = np.loadtxt("D2a.txt")
dataB = np.loadtxt("D2b.txt")
dataZ = np.loadtxt("D2z.txt")


# In[82]:


def predict(x,data):
    
    num_features = data.shape[1] - 1
    num_points = data.shape[0]
    
    min_dists = np.array([[math.inf, -1],[math.inf, -1],[math.inf, -1]])
    
    for i in range(num_points):
        
        dist = distance(x,data[i,0:num_features])
        
        # print(data[i,0:num_features])
        
        found = 0
        for j in range(3):
            if dist < min_dists[j,0] and found == 0:
                min_dists[j,0] = dist
                min_dists[j,1] = data[i,num_features]
                found = 1
    
    # print(min_dists)
    
    count_ones = 0
    for i in range(3):
        if min_dists[i,1] == 1:
            count_ones += 1
    
    if count_ones > 1:
        return 1
    else:
        return 0


# In[81]:


predict([1,1],dataB)


# In[109]:


def plot_data(data):
    i = 0
    onesx = np.empty((0,2))
    onesy = np.empty((0,2))
    zerosx = np.empty((0,2))
    zerosy = np.empty((0,2))
    for y in data[:,2]:
        if y == 1:
            onesx = np.append(onesx, data[i,0])
            onesy = np.append(onesy, data[i,1])
        else:
            zerosx = np.append(zerosx, data[i,0])
            zerosy = np.append(zerosy, data[i,1])
        i += 1
    plt.plot(onesx, onesy, 'rd', label="y=1")
    plt.plot(zerosx, zerosy, 'bs', label="y=0")
    plt.legend()
    
    plt.xlim(-2.5,2.5)
    plt.ylim(-2.5,2.5)


# In[112]:


# question 3.8
# points with x1 on -2, -1.9, ... , 2 and x2 on the same range, independently
# plot predictions overlayed with the training set

test_set = []

i = -2.0
while i < 2.1:
    j = -2.0
    while j < 2.1:
        test_set.append(np.array([i,j,-1]))
        j += 0.1
    i += 0.1
test_set = np.asarray(test_set)

# print(test_set)

i = 0
for row in test_set:
    
    # print(row[0:2])    
    test_set[i,2] = predict(row[0:2], dataZ)
    i += 1

plot_data(test_set)


# In[113]:


plot_data(dataZ)


# In[124]:


# question 3.9
# build 4 folds and complements from dataA

fold_indices = random.sample(range(200),200)
data_1 = np.empty((0,7))
data_1_comp = np.empty((0,7))
data_2 = np.empty((0,7))
data_2_comp = np.empty((0,7))
data_3 = np.empty((0,7))
data_3_comp = np.empty((0,7))
data_4 = np.empty((0,7))
data_4_comp = np.empty((0,7))

data = dataA

j = 0
for i in fold_indices:
    if j < 50:
        data_1 = np.append(data_1, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
        data_2_comp = np.append(data_2_comp, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
        data_3_comp = np.append(data_3_comp, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
        data_4_comp = np.append(data_4_comp, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
    if j >= 50 and j < 100:
        data_2 = np.append(data_2, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
        data_1_comp = np.append(data_1_comp, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
        data_3_comp = np.append(data_3_comp, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
        data_4_comp = np.append(data_4_comp, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
    if j >= 100 and j < 150:
        data_3 = np.append(data_3, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
        data_1_comp = np.append(data_1_comp, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
        data_2_comp = np.append(data_2_comp, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
        data_4_comp = np.append(data_4_comp, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
    if j >= 150 and j < 200:
        data_4 = np.append(data_4, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
        data_1_comp = np.append(data_1_comp, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
        data_2_comp = np.append(data_2_comp, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
        data_3_comp = np.append(data_3_comp, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
    j += 1

# print(data_1.shape[0])
# print(data_1.shape[1])
# print(data_1_comp.shape[0])
# print(data_1_comp.shape[1])


# In[164]:


# need to run this and the 3 snippets below each time the data variable is updated

count_error = 0
count = 0
for row in data_1:
    if row[6] != predict(row[0:6], data_1_comp):
        count_error += 1
    count += 1

print("error:", count_error * 100 / count, "%")


# In[155]:


count_error = 0
count = 0
for row in data_2:
    if row[6] != predict(row[0:6], data_2_comp):
        count_error += 1
    count += 1

print("error:", count_error * 100 / count, "%")


# In[156]:


count_error = 0
count = 0
for row in data_3:
    if row[6] != predict(row[0:6], data_3_comp):
        count_error += 1
    count += 1

print("error:", count_error * 100 / count, "%")


# In[157]:


count_error = 0
count = 0
for row in data_4:
    if row[6] != predict(row[0:6], data_4_comp):
        count_error += 1
    count += 1

print("error:", count_error * 100 / count, "%")


# In[158]:


# normalize D2a.txt

data = dataA

# print("dataA")
# print(dataA)

means = []
stdvs = []

for i in range(7):
    means.append(np.mean(data[:,i]))
    stdvs.append(np.std(data[:,i]))

# print("before")
# print(data)

for i in range(200):
    for j in range(6):
        data[i,j] = (data[i,j] - means[j]) / stdvs[j]

# print("after")
# print(data)

print(means)
print(stdvs)


# In[153]:


# question 3.9
# build 4 folds and complements from dataA

fold_indices = random.sample(range(200),200)
data_1 = np.empty((0,7))
data_1_comp = np.empty((0,7))
data_2 = np.empty((0,7))
data_2_comp = np.empty((0,7))
data_3 = np.empty((0,7))
data_3_comp = np.empty((0,7))
data_4 = np.empty((0,7))
data_4_comp = np.empty((0,7))

j = 0
for i in fold_indices:
    if j < 50:
        data_1 = np.append(data_1, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
        data_2_comp = np.append(data_2_comp, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
        data_3_comp = np.append(data_3_comp, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
        data_4_comp = np.append(data_4_comp, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
    if j >= 50 and j < 100:
        data_2 = np.append(data_2, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
        data_1_comp = np.append(data_1_comp, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
        data_3_comp = np.append(data_3_comp, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
        data_4_comp = np.append(data_4_comp, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
    if j >= 100 and j < 150:
        data_3 = np.append(data_3, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
        data_1_comp = np.append(data_1_comp, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
        data_2_comp = np.append(data_2_comp, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
        data_4_comp = np.append(data_4_comp, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
    if j >= 150 and j < 200:
        data_4 = np.append(data_4, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
        data_1_comp = np.append(data_1_comp, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
        data_2_comp = np.append(data_2_comp, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
        data_3_comp = np.append(data_3_comp, [[data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6]]], axis=0)
    j += 1

# print(data_1.shape[0])
# print(data_1.shape[1])
# print(data_1_comp.shape[0])
# print(data_1_comp.shape[1])


# In[178]:


# question 3.10

fold_indices = random.sample(range(200),200)
data_1 = np.empty((0,3))
data_1_comp = np.empty((0,3))
data_2 = np.empty((0,3))
data_2_comp = np.empty((0,3))
data_3 = np.empty((0,3))
data_3_comp = np.empty((0,3))
data_4 = np.empty((0,3))
data_4_comp = np.empty((0,3))

# data = dataB

j = 0
for i in fold_indices:
    if j < 50:
        data_1 = np.append(data_1, [[data[i,0], data[i,1], data[i,2]]], axis=0)
        data_2_comp = np.append(data_2_comp, [[data[i,0], data[i,1], data[i,2]]], axis=0)
        data_3_comp = np.append(data_3_comp, [[data[i,0], data[i,1], data[i,2]]], axis=0)
        data_4_comp = np.append(data_4_comp, [[data[i,0], data[i,1], data[i,2]]], axis=0)
    if j >= 50 and j < 100:
        data_2 = np.append(data_2, [[data[i,0], data[i,1], data[i,2]]], axis=0)
        data_1_comp = np.append(data_1_comp, [[data[i,0], data[i,1], data[i,2]]], axis=0)
        data_3_comp = np.append(data_3_comp, [[data[i,0], data[i,1], data[i,2]]], axis=0)
        data_4_comp = np.append(data_4_comp, [[data[i,0], data[i,1], data[i,2]]], axis=0)
    if j >= 100 and j < 150:
        data_3 = np.append(data_3, [[data[i,0], data[i,1], data[i,2]]], axis=0)
        data_1_comp = np.append(data_1_comp, [[data[i,0], data[i,1], data[i,2]]], axis=0)
        data_2_comp = np.append(data_2_comp, [[data[i,0], data[i,1], data[i,2]]], axis=0)
        data_4_comp = np.append(data_4_comp, [[data[i,0], data[i,1], data[i,2]]], axis=0)
    if j >= 150 and j < 200:
        data_4 = np.append(data_4, [[data[i,0], data[i,1], data[i,2]]], axis=0)
        data_1_comp = np.append(data_1_comp, [[data[i,0], data[i,1], data[i,2]]], axis=0)
        data_2_comp = np.append(data_2_comp, [[data[i,0], data[i,1], data[i,2]]], axis=0)
        data_3_comp = np.append(data_3_comp, [[data[i,0], data[i,1], data[i,2]]], axis=0)
    j += 1

# print(data_1.shape[0])
# print(data_1.shape[1])
# print(data_1_comp.shape[0])
# print(data_1_comp.shape[1])


# In[179]:


# need to run this and the 3 snippets below each time the data variable is updated

count_error = 0
count = 0
for row in data_1:
    if row[2] != predict(row[0:2], data_1_comp):
        count_error += 1
    count += 1

print("error:", count_error * 100 / count, "%")


# In[180]:


count_error = 0
count = 0
for row in data_2:
    if row[2] != predict(row[0:2], data_2_comp):
        count_error += 1
    count += 1

print("error:", count_error * 100 / count, "%")


# In[181]:


count_error = 0
count = 0
for row in data_3:
    if row[2] != predict(row[0:2], data_3_comp):
        count_error += 1
    count += 1

print("error:", count_error * 100 / count, "%")


# In[182]:


count_error = 0
count = 0
for row in data_4:
    if row[2] != predict(row[0:2], data_4_comp):
        count_error += 1
    count += 1

print("error:", count_error * 100 / count, "%")


# In[177]:


# normalize D2b.txt

data = dataB

print("dataB")
print(dataB)

means = []
stdvs = []

for i in range(3):
    means.append(np.mean(data[:,i]))
    stdvs.append(np.std(data[:,i]))

print("before")
print(data)

for i in range(200):
    for j in range(2):
        data[i,j] = (data[i,j] - means[j]) / stdvs[j]

print("after")
print(data)

print(means)
print(stdvs)


# In[ ]:




