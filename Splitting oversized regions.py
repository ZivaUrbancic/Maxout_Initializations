#!/usr/bin/env python
# coding: utf-8

# In[53]:


# get_ipython().run_line_magic('matplotlib', 'notebook')

import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import cm
from scipy.linalg import null_space
from scipy.spatial.distance import cdist, euclidean


exec(open("mnist.py").read())

#np.random.seed(64433)


# # Creating data

# In[178]:

X = np.load("MNISTprojected.npy")
Xlabels = np.load("MNISTlabels.npy")



d = 3      # dimension of the ambient space
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

X3d = np.concatenate((X, X1), axis = 1)[:,:-1]
X = X3d

N *= 10

xmax = np.max(X[:,0])
xmin = np.min(X[:,0])
ymax = np.max(X[:,1])
ymin = np.min(X[:,1])
zmax = np.max(X[:,2])
zmin = np.min(X[:,2])

#xmax = np.max(X[:,0])
#xmin = np.min(X[:,0])
#ymax = np.max(X[:,1])
#ymin = np.min(X[:,1])

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


# In[349]:


def marginal_median(Y):
    '''
    Find the marginal median of a data set Y.

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

# copied from https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points

def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1

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
    
    Y = np.array(Y)
    #print(Y.shape)
    
    assert Y.shape[1] >= Y.shape[0], 'More data points than dimensions'
    
    d = Y.shape[0]
    matrix = np.concatenate((Y, np.ones((d,1))), axis = 1)
    
    return null_space(matrix).transpose()[0]
    


# In[234]:

def regions_matrix(N, m):
    R = np.concatenate((np.zeros((N,m), dtype=int),
                    np.arange(N).reshape(-1,1)),
                   axis = 1)
    return R

R = regions_matrix(N, m)

# In[235]:


def linear(weights_and_bias):
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
        return np.dot(x,weights_and_bias[:-1]) + weights_and_bias[-1]
    return f

def linear_list(weights_and_biases):
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
    for w in weights_and_biases:
        l += [linear(w)]
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

def update_region_array(X, R, functions, c):
    
    sorted_X = X[R[:,-1]]
    regs = regions(sorted_X, functions)
    r = R.copy()
    r[:,c] = regs
    sorted_indices = np.lexsort(np.rot90(r))

    return r[sorted_indices]


# In[239]:



# In[238]:


#f1 = linear([1,0,0])
#f2 = linear([0,1,0])
#f3 = linear([0,0,0])

#R_ = update_region_array(X, R, [f1,f2,f3], 2)



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

def top_k_median_points(ordered_indices, X,
                        k = -1,
                        marginal = False):
    '''
    Compute the medians of the largest k regions as given by ordered_indices.

    Parameters
    ----------
    ordered_indices : List
        List of lists of indices, ordered by group size.
    X : Array
        Data set.
    k : Int
        Number of regions whose medians will be computed.

    Returns
    -------
    medians : List
        List of medians of the k largest regions.

    '''
    
    if k == -1:
        k = X.shape()[1]

    medians = []

    for i in range(k):
        indices = ordered_indices[i]
        data = X[indices]
        #print(data, marginal_median(data), '\n\n\n')
        if marginal:
            medians += [marginal_median(data)]
        else:
            medians += [geometric_median(data)]
    
    return medians

def hyperplane_through_medians(ordered_indices, X,
                               marginal = False):
    '''
    Create hyperplane through the medians of the largest regions
    (as many as possible).

    Parameters
    ----------
    ordered_indices : List
        List of indices.
    X : Array
        Data set.

    Returns
    -------
    Array
        Normal vector to the hyperplane in R^{d+1}.
     
    '''
    
    k = min(X.shape[1], len(ordered_indices))
    points = top_k_median_points(ordered_indices, X, k, marginal)
    
    #points_plot = np.array(points)
    #ax.scatter(points_plot[:,0], points_plot[:,1], c = 'r')
    
    return hyperplane_through_points(points)

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

R = np.concatenate((np.zeros((N,m), dtype=int),
                np.arange(N).reshape(-1,1)),
               axis = 1)

def zero(x):
    return 0


def initialise_layer(X, m):
    
    N, d = X.shape
    W = []
    R = regions_matrix(N, m)
    
    for k in range(m):
        indices = find_ordered_region_indices(R)
        w = hyperplane_through_medians(indices, X)
        f = linear(w)
        R = update_region_array(X, R, [f,zero], k)
        W += [w]
    
    W = np.array(W)
    
    return np.array(W)




region_cardinalities = [[],[]]
if d == 2:
    for mar in range(1):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(X[:,0], X[:,1], c = 'black')
        ax.set_ylim(ymin, ymax)
        
        W = []
        
        x = np.array([xmin,xmax])
        for k in range(m):
            indices = find_ordered_region_indices(R)
            w = hyperplane_through_medians(indices, X, mar)
            f = linear(w)
            R = update_region_array(X, R, [f,zero], k)
            W += [w]
            
            wx = -w[0]/w[1]
            wc = -w[2]/w[1]
            y = wx*x + wc
            
            if k == m-1:
                indices = find_ordered_region_indices(R)
                for i in indices:
                    region_cardinalities[mar] += [len(i)]
                    ax.scatter(X[i,0], X[i,1])
            
            print('Done', k)
            ax.plot(x,y, c = 'gray')



# In[ ]:

#fig, ax = plt.subplots()
#ax.plot(region_cardinalities[0], c = 'green')
#ax.plot(region_cardinalities[1], c = 'magenta')

#print(np.std(region_cardinalities[0]),
#      np.std(region_cardinalities[1]))
