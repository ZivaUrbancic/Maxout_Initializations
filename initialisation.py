#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 13:33:28 2021

@author: ziva
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import cm
from scipy.linalg import null_space
from scipy.spatial.distance import cdist, euclidean


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



def initialise_region_matrix(N):
    #R = np.concatenate((np.zeros((N,m), dtype=int),
    #                np.arange(N).reshape(-1,1)),
    #               axis = 1)
    return np.arange(N).reshape(-1,1)



def zero(x):
    return 0



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



def update_region_matrix(X, R, functions):
    
    sorted_X = X[R[:,-1]]
    regs = regions(sorted_X, functions)
    r = R.copy()
    #print(r[:,-1])
    indices=R[:, -1].reshape(-1,1)
    #print(indices)
    r[:,-1] = regs
    r = np.concatenate((r, indices), axis=1)
    sorted_indices = np.lexsort(np.rot90(r))

    return r[sorted_indices]




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




def initialise_layer(X, R, m):
    '''
    Initializes a neural network layer.

    Parameters
    ----------
    ordered_indices : List
        List of indices.
    X : Array
        Data set.
    R : Array
        Region matrix.
    m : Int
        Width of next layer

    Returns
    -------
    Array
        A neural network layer initialized with the weights and biases computed
        with scripts before.
    '''
    n = X.shape[1]
    f=nn.Linear(n, m)
    W=[]
    for k in range(m):
            indices = find_ordered_region_indices(R)
            w = hyperplane_through_medians(indices, X)
            R = update_region_matrix(X, R, [linear(w),zero])
            W += [w]
    W = np.array(W)
    Weights = W[:,:-1]
    Biases = W[:,-1]
    with torch.no_grad():
        f.weight = nn.Parameter(torch.tensor(Weights,dtype=torch.float32))
        f.bias = nn.Parameter(torch.tensor(Biases,dtype=torch.float32))
    print(W)
    print(torch.tensor(Biases, dtype=torch.float64))
    return f



def fix_variance(weights, biases):
    m, n = weights.shape
    scale_factor = np.sqrt(2/n) / np.std(weights)
    return scale_factor*weights, scale_factor*biases
    



def initialise_ReLU_network(model,X):
    Layers = [layer for layer in model.children()]
    R = initialise_region_matrix(X.shape[0])
    for layer in Layers:
        W=[]
        for k in range(layer.in_features):
                indices = find_ordered_region_indices(R)
                w = hyperplane_through_medians(indices, X)
                R = update_region_matrix(X, R, [linear(w),zero])
                W += [w]
        W = np.array(W)
        Weights = W[:,:-1]
        Biases = W[:,-1]
        Weights, Biases = fix_variance(Weights,Biases)
        layer.weight = Weights
        layer.bias = Biases
        X = layer(X)
        
# =============================================================================
#         with torch.no_grad():
#             f.weight = nn.Parameter(torch.tensor(Weights,dtype=torch.float32))
#             f.bias = nn.Parameter(torch.tensor(Biases,dtype=torch.float32))
# =============================================================================
        