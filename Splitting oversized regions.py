#!/usr/bin/env python
# coding: utf-8

# In[53]:


# get_ipython().run_line_magic('matplotlib', 'notebook')

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space

np.random.seed(64432)


# # Creating data

# In[178]:


d = 2      # dimension of the ambient space
N = 30     # size of the data set
k = 3 
m = 6    # number of units

X1 = np.random.normal(0, 1, (N, d))
X2 = np.random.normal((5,1), 1, (N, d))
X3 = np.random.normal((0,5), 1, (N, d))

N *= 3

X = np.concatenate((X1,X2,X3))

xs = X[:, 0]
ys = X[:, 1]

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(xs, ys)


# # Sample initial weight

# In[19]:


weight_0 = np.random.random(d)
weight_0


# In[349]:


def median_point(Y):
    '''
    Find the median point of a data set Y.

    Parameters
    ----------
    Y : List or Array
        Data set in R^n.

    Returns
    -------
    Array
        Component-wise median point in R^n.

    '''
    
    d= Y.shape[1]
    point = []
    
    for n in range(d):
        point += [np.median(Y[:,n])]
        
    return np.array(point)


# In[72]:


def hyperplane_through_points(Y):
    '''
    Find a hyperplane which goes through a collection of up to
    n points Y in R^n.

    Parameters
    ----------
    Y : List or Array
        Y is the (up to) n points in R^n.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    # Y is a list (or array) of d points in R^d
    
    d = Y[0].shape[0]
    matrix = np.concatenate((Y, np.ones((d,1))), axis = 1)
    
    return null_space(matrix).transpose()[0]
    


# In[234]:


R = np.concatenate((np.zeros((N,m), dtype=int),
                    np.arange(N).reshape(-1,1)),
                   axis = 1)


# In[235]:


def linear(weight, bias):
    '''
    Create affine linear function f: R^n -> R with given weights and bias.

    Parameters
    ----------
    weight : List or Array
        Weights for each component of R^n.
    bias : Float
        Constant term.

    Returns
    -------
    Function R^n -> R
        Affine linear function.

    '''
    def f(x):
        return np.dot(x,weight) + bias
    return f

def linear_list(weight, biases):
    '''
    Create a list of affine linear functions R^n -> R with given
    weights and biases.

    Parameters
    ----------
    weight : List or Array of Arrays
        Weights for each function.
    biases : TYPE
        Biases for each function.

    Returns
    -------
    l : List of functions
        Affine linear functions.

    '''
    l = []
    for bias in biases:
        l += [linear(weight, bias)]
    return l

def regions(X, functions):
    '''
    Split the data set X into regions by which function is maximal
    at each point in X.

    Parameters
    ----------
    X : List or Array
        Data set to be divided.
    functions : List of functions
        Functions to determine the regions.

    Returns
    -------
    reg : List
        List of the index of the function that was maximal at
        each point x in X.

    '''

    reg = []
    
    for x in X:
        vals = []
        for function in functions:
            vals += [function(x)]
        reg += [np.argmax(vals)]
    
    return reg

def update_region_array(R, regs, c):
    '''
    Updates the regions matrix R with the new regions information
    by inserting the list regs in the c^th column.

    Parameters
    ----------
    R : Array
        Current regions matrix.
    regs : List or Array
        New column of region indices.
    c : Int
        Index of the column to be updated.

    Returns
    -------
    Array
        New regions matrix.

    '''
    
    r = R.copy()

    r[:,c] = regs
    np.sort(r,0)

    
    return r[np.lexsort(np.rot90(r))]


# In[239]:



# In[238]:


f1 = linear([1,0],0)
f2 = linear([0,1],0)
f3 = linear([0,0],0)

R_ = update_region_array(R, regions(X, [f1,f2,f3]), 2)

R_


# In[ ]:





# In[337]:


def find_ordered_region_indices(R):
    '''
    Group together the data in X by region, and return a sorted list of
    regions, each described by a list of the indices of the points inside it.
    Regions are sorted largest to smallest, so return[:k] are the k largest.

    Parameters
    ----------
    R : Array
        Regions matrix.

    Returns
    -------
    List
        DESCRIPTION.

    '''
    
    # Initialise lists of elements of each group and the size of each group.
    region_groups = [[R[0,-1]]]
    region_sizes = [1]
    
    # Add a data point to the current group if it shares the same region
    # signature as the previous point, and create a new group if not. We
    # count sizes as we go.
    current_region_signature = R[0,:-1]
    for n in range(1,N):
        if np.array_equal(R[n,:-1], current_region_signature):
            region_groups[-1] += [R[n,-1]]
            region_sizes[-1] += 1
        else:
            region_groups += [[R[n,-1]]]
            region_sizes += [1]
            current_region_signature = R[n,:-1]
    
    # Sort region indices by size (and reverse to sort largest -> smallest)
    sorted_region_indices = np.argsort(region_sizes)[::-1]
    
    return [region_groups[i] for i in sorted_region_indices]

def top_d_median_points(ordered_indices, d, X):
    '''
    Compute the medians of the largest d regions as given by ordered_indices.

    Parameters
    ----------
    ordered_indices : List
        List of lists of indices, ordered by group size.
    d : Int
        Number of regions whose medians will be computed.
    X : Array
        Data set.

    Returns
    -------
    medians : List
        List of medians of the d largest regions.

    '''

    medians = []

    for i in range(d):
        indices = ordered_indices[i]
        data = X[indices]
        medians += [median_point(data)]

    return medians


# In[350]:


top_d_median_points(find_ordered_region_indices(R_), d, X)

r = find_ordered_region_indices(R_)
meds = top_d_median_points(r, d, X)

fig = plt.figure()
ax = fig.add_subplot()#projection='2d')

for group in r:
    XX = X[group]
    xs = XX[:,0]
    ys = XX[:,1]
    ax.scatter(xs, ys)
    
for p in meds:
    ax.scatter(p[0],p[1], c = 'black')
    

meds


# In[262]:

    

# In[ ]:




