#!/usr/bin/env python
# coding: utf-8

# In[53]:


# get_ipython().run_line_magic('matplotlib', 'notebook')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import cm
from scipy.linalg import null_space
from scipy.spatial.distance import cdist, euclidean


exec(open("mnist.py").read())
exec(open("initialisation.py").read())

#np.random.seed(64433)


# # Creating data

# In[178]:

X = np.load("MNISTprojected.npy")
Xlabels = np.load("MNISTlabels.npy")



d = 2      # dimension of the ambient space
N = 20     # size of the data set
k = 3 
m = 20    # number of units

#X1 = np.random.normal(0, 1, (N, d))
#X2 = np.random.normal((4,3), 1, (N, d))
#X3 = np.random.normal((0,5), 1, (N, d))
#N *= 3
#X = np.concatenate((X1,X2,X3))

X, Xlabels = load_MNIST_projected(X,Xlabels,N, True)
X1, Xlabels1 = load_MNIST_projected(X,Xlabels,N, True)

#X3d = np.concatenate((X, X1), axis = 1)[:,:-1]
#X = X3d

N *= 10

# =============================================================================
# xmax = np.max(X[:,0])
# xmin = np.min(X[:,0])
# ymax = np.max(X[:,1])
# ymin = np.min(X[:,1])
# zmax = np.max(X[:,2])
# zmin = np.min(X[:,2])
# 
# =============================================================================
xmax = np.max(X[:,0])
xmin = np.min(X[:,0])
ymax = np.max(X[:,1])
ymin = np.min(X[:,1])

xs = X[:, 0]
ys = X[:, 1]

fig = plt.figure()
ax = fig.add_subplot()
colours = ['r', #0
           'g', #1
           'b', #2
           'yellow', #3
           'black', #4
           'magenta', #5
           'gray', #6
           'cyan', #7
           'orange', #8
           'purple'] #9
for k in range(N):
    I = Xlabels[k]
    ax.scatter(xs[k], ys[k], c = colours[I])


# # Sample initial weight

# In[19]:


weight_0 = np.random.random(d)
weight_0

R = initialise_region_matrix(N)

# In[235]:




# In[239]:



# In[238]:
# =============================================================================
# R = initialise_region_matrix(N)
# 
# f1 = linear([1,0,0])
# f2 = linear([0,1,0])
# f3 = linear([0,0,0])
# 
# print(R)
# update_region_matrix(X, R, [f1,f2,f3])
# =============================================================================

# In[337]:



# In[350]:

#r = find_ordered_region_indices(R_)
#meds = top_k_median_points(r, X, d)

#fig = plt.figure()
#ax = fig.add_subplot()

#for group in r:
#    XX = X[group]
#    xs = XX[:,0]
#    ys = XX[:,1]
    #ax.scatter(xs, ys)
    
#for p in meds:
    #ax.scatter(p[0],p[1], c = 'black')

#x = np.array([-3,8])
#w = hyperplane_through_medians(r, X)

#wx = -w[0]/w[1]
#wc = -w[2]/w[1]
#y = wx*x + wc

#ax.plot(x,y, c = 'gray')

# In[262]:





# =============================================================================
# R = np.concatenate((np.zeros((N,m), dtype=int),
#                 np.arange(N).reshape(-1,1)),
#                axis = 1)
# 
# region_cardinalities = [[],[]]
# if d == 2:
#     for mar in range(1):
#         fig = plt.figure()
#         ax = fig.add_subplot()
#         ax.scatter(X[:,0], X[:,1], c = 'black')
#         ax.set_ylim(ymin, ymax)
#         
#         W = []
#         
#         x = np.array([xmin,xmax])
#         for k in range(m):
#             indices = find_ordered_region_indices(R)
#             w = hyperplane_through_medians(indices, X, mar)
#             f = linear(w)
#             R = update_region_matrix(X, R, [f,zero])
#             W += [w]
#             
#             wx = -w[0]/w[1]
#             wc = -w[2]/w[1]
#             y = wx*x + wc
#             
#             if k == m-1:
#                 indices = find_ordered_region_indices(R)
#                 for i in indices:
#                     region_cardinalities[mar] += [len(i)]
#                     ax.scatter(X[i,0], X[i,1])
#             
#             print('Done', k)
#             ax.plot(x,y, c = 'gray')
# =============================================================================



    

R = initialise_region_matrix(N)
f = initialise_layer(X, R, m)        

# In[ ]:

#fig, ax = plt.subplots()
#ax.plot(region_cardinalities[0], c = 'green')
#ax.plot(region_cardinalities[1], c = 'magenta')

#print(np.std(region_cardinalities[0]),
#      np.std(region_cardinalities[1]))
