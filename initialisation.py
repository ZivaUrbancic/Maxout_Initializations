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
def geometric_median(Xin, eps=1e-5):

    X = np.array(Xin)
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
            return torch.tensor(y)
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return torch.tensor(y1)
        y = y1



def hyperplane_through_points(Yin):
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

    Y = np.array([np.array(vector) for vector in Yin])


    assert Y.shape[1] >= Y.shape[0], 'More data points than dimensions'

    d = Y.shape[0]
    matrix = np.concatenate((Y, np.ones((d,1))), axis = 1)

    null = null_space(matrix)
    random_vector = np.random.randn(null.shape[1])

    return np.matmul(null, random_vector) + 0.0001*np.random.randn(null.shape[0])



def initialise_region_matrix(N):
    return np.arange(N).reshape(-1,1)

def initialise_costs_vector(N):
    costs = -np.ones(N)
    costs[0] = 1
    return costs

def initial_sample(Xall, Yall, n_samples_per_label=1):
    Xall_indices = np.arange(Xall.shape[0])
    np.random.shuffle(Xall_indices)
    label_counter = np.zeros(10) # for MNIST only!!!
    X = []
    Y = []
    R = []
    for i in Xall_indices:
        if label_counter[Yall[i]]<n_samples_per_label:
            X += [Xall[i]]
            Y += [Yall[i]]
            R += [[i]]
            label_counter[Yall[i]] += 1
        if len(Y) == n_samples_per_label*10:
            break

    R = np.array(R)
    C = initialise_costs_vector(len(Y))

    return torch.stack(X), Y, R, C



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



def regions(X, preactivation_unit):
    '''
    Split the data set X into regions by which func
tion is maximal
    at each point in X.

    Parameters
    ----------
    X : List or Array
        Data set to be divided.
    preactivation_unit : List of preactivation_unit
        Preactivation_Unit to determine the regions.

    Returns
    -------
    reg : List
        List of the index of the function that was maximal at
        each point x in X.

    '''

    reg = []

    for x in X:
        vals = []
        for function in preactivation_unit:
            vals += [function(x)]
        reg += [np.argmax(vals)]

    return reg


def vector_variance(Y):

    mean = np.mean(Y, axis = 0)
    differences = Y - mean
    var = np.sum(differences**2)/Y.shape[0]

    return var

def L2_region_cost(indices, Y):
    # indices of a region
    # Y is the whole label distribution

    region_labels = np.array(Y[indices])
    N = len(indices)

    return N*vector_variance(region_labels)

def regions_from_costs(costs):

    region_start_points = []
    for n, c in enumerate(costs):
        if c >= 0:
            region_start_points += [n]
    region_start_points += [-1]

    return [[region_start_points[i],
             region_start_points[i+1]]
            for i in range(len(region_start_points) - 1)]

def regions_from_matrix(R):

    region_start_points = [0]
    current_region_signature = R[0,:-1]

    for n in range(1,R.shape[0]):
        if not np.array_equal(R[n,:-1], current_region_signature):
            region_start_points += [n]
            current_region_signature = R[n,:-1]
    region_start_points += [-1]

    return [[region_start_points[i],
             region_start_points[i+1]]
            for i in range(len(region_start_points) - 1)]

def update_regions_and_costs(R, C, functions, X, Y, region_cost):

    indices = R[:, -1] # save indices column of R
    before_regions = regions_from_costs(C) # save old regions

    regs = np.array(regions(X[indices], functions)) # compute new regions column
    R = np.concatenate((R[:,:-1],
                        regs.reshape(-1,1),
                        indices.reshape(-1,1)),
                       axis=1) # construct new R

    sorted_row_indices = np.lexsort(np.rot90(R))
    R = R[sorted_row_indices] # sort region matrix
    C = C[sorted_row_indices] # sort cost vector

    after_regions = regions_from_matrix(R) # new regions

    i = 0
    for region in after_regions:

        if not region == before_regions[i]:
            if region[1] == -1:
                data_indices = R[region[0]:, -1]
            else:
                data_indices = R[region[0] : region[1], -1]
            C[region[0]] = region_cost(data_indices, Y)

        if region[1] == before_regions[i][1]:
            i += 1

    C_sorted = np.sort(C)[::-1];
    k = min(X.shape[1], len(after_regions))
    if (C_sorted[k-1]<1):
        need_resampling = True
    else:
        need_resampling = False

    return R, C, need_resampling

# def find_ordered_region_indices(R):
#     '''
#     Group together the data in X by region, and return a sorted list of
#     regions, each described by a list of the indices of the points inside it.
#     Regions are sorted largest to smallest, so return[:k] are the k largest.

#     Parameters
#     ----------
#     R : Array
#         Regions matrix.

#     Returns
#     -------
#     List
#         DESCRIPTION.

#     '''

#     # Initialise lists of elements of each group and the size of each group.
#     region_groups = [[R[0,-1]]]
#     region_sizes = [1]

#     # Add a data point to the current group if it shares the same region
#     # signature as the previous point, and create a new group if not. We
#     # count sizes as we go.
#     current_region_signature = R[0,:-1]
#     for n in range(1,R.shape[0]):
#         if np.array_equal(R[n,:-1], current_region_signature):
#             region_groups[-1] += [R[n,-1]]
#             region_sizes[-1] += 1
#         else:
#             region_groups += [[R[n,-1]]]
#             region_sizes += [1]
#             current_region_signature = R[n,:-1]

#     # Sort region indices by size (and reverse to sort largest -> smallest)
#     sorted_region_indices = np.argsort(region_sizes)[::-1]


#     return [region_groups[i] for i in sorted_region_indices]


def hyperplane_through_largest_regions(X, R, C,
                               marginal = False):

    regions = regions_from_costs(C)
    costs = C[[pair[0] for pair in regions]]

    sorted_region_indices = np.argsort(costs)[::-1]

    k = min(X.shape[1], len(sorted_region_indices))

    medians = []

    for i in range(k):
        matrix_indices = regions[sorted_region_indices[i]]
        if matrix_indices[0] == R.shape[0] - 1:
            data_indices = [R[-1, -1]]
        else:
            data_indices = R[matrix_indices[0] : matrix_indices[1], -1]
        print(data_indices)
        data = X[data_indices]
        if marginal:
            medians += [marginal_median(data)]
        else:
            medians += [geometric_median(data)]

    return hyperplane_through_points(medians)


def resample(Xall, Yall, X, Y, R, C, preactivation_):

    # construct list of indices of points in Xall that are not in X
    Xextra_indices = [i for i in range(len(Xall)) if i not in R[:,-1]]
    Xextra_indices = np.shuffle(Xextra_indices)

    # double sample size by sampling new points
    Xextra = []
    Yextra = []
    for i in len(X):
      Xextra += Xall[Xextra_indices[i]]
      Yextra += Yall[Xextra_indices[i]]

    # construct unsorted region matrix of new points
    Rextra_columns = []
    for preactivation_unit in preactivation_functions:
        Rextra_columns += [regions(Xextra,preactivation_unit).reshape(-1,1)]
    Rextra_columns += [Xextra_indices.reshape(-1,1)]
    Rextra = np.concatenate(Rextra_columns,axis=1)

    # update X, Y, R
    X = np.concatenate([X,Xextra],axis=0)
    Y = np.concatenate([Y,Yextra],axis=0)
    R = np.concatenate([R,Rextra],axis=0)
    R = R[np.lexsort(np.rot90(R))]

    # recompute region costs
    regions = regions_from_matrix(R)
    C = -np.ones(len(R))
    for region in regions:
        if region[1] == -1:
            data_indices = R[region[0]:, -1]
        else:
            data_indices = R[region[0] : region[1], -1]
        C[region[0]] = region_cost(data_indices, Y)

    return X, Y, R, C

# def initialise_layer(X, Y, R, m):
#     '''
#     Initializes a neural network layer.

#     Parameters
#     ----------
#     ordered_indices : List
#         List of indices.
#     X : Array
#         Data set.
#     R : Array
#         Region matrix.
#     m : Int
#         Width of next layer

#     Returns
#     -------
#     Array
#         A neural network layer initialized with the weights and biases computed
#         with scripts before.
#     '''
#     n = X.shape[1]
#     f=nn.Linear(n, m)
#     W=[]
#     for k in range(m):
#         w = hyperplane_through_largest_regions(X, R)
#         update_region_matrix(R, [linear(w),zero], X, Y, L2_region_cost)
#         W += [w]
#     W = np.array(W)
#     Weights = W[:,:-1]
#     Biases = W[:,-1]
#     with torch.no_grad():
#         f.weight = nn.Parameter(torch.tensor(Weights,dtype=torch.float32))
#         f.bias = nn.Parameter(torch.tensor(Biases,dtype=torch.float32))
#     #print(W)
#     #print(torch.tensor(Biases, dtype=torch.float64))
#     return f



def fix_variance(weights, biases):
    m, n = weights.shape
    scale_factor = (2/n)**0.5 / torch.std(weights)
    return scale_factor*weights, scale_factor*biases




def reinitialise_ReLU_network(model, Xall, Yall):
    if 'DEBUG_PRINT' not in globals():
        DEBUG_PRINT = 0

    Layers = [layer for layer in model.children()]

    ###
    # Initialization:
    #   - X sample of Xall
    #   - Y labels of X
    #   - R initial region matrix, one column vector with indices of X
    #   - C initial cost vector
    ###
    if DEBUG_PRINT>0:
        print("constructing initial sample...\n")
    X, Y, R, C = initial_sample(Xall, Yall)

    layer_counter = 0
    preactivation_functions = []

    for layer in Layers: # iterate through the layers
        W = []
        if DEBUG_PRINT>0:
            layer_counter += 1
            print("initialising layer ",layer_counter," of ",len(Layers),"...\n")

        for k in range(layer.out_features): # iterate through the maxout units within a layer
            if DEBUG_PRINT>1:
                print("initialising unit ",k," of ",layer.out_features,"...\n")
            w = hyperplane_through_largest_regions(X, R, C) # construct a linear region boundary
            preactivation_unit = [linear(w),zero]
            R, C, need_resampling = update_regions_and_costs(R, C, preactivation_unit, X, Y, L2_region_cost) # update region matrix and cost vector, check whether resampling necessary
            preactivation_functions += [preactivation_unit]
            W += [w]
            if need_resampling:
                if DEBUG_PRINT>1:
                    print("resampling. increasing sample size from ",len(X)," to ",2*len(X),"...\n")
                X, Y, R, C  = resample(Xall, Yall, X, Y, R, C, preactivation_functions) # resample X if necessary
        W = torch.tensor(W, dtype = torch.float32)
        Weights = W[:,:-1]
        Biases = W[:,-1]
        Weights, Biases = fix_variance(Weights,Biases) # control variance of weights and biases without changing linear region boundary
        layer.weight = nn.Parameter(Weights)
        layer.bias = nn.Parameter(Biases)
        with torch.no_grad():
            X = layer(X)
        break


# =============================================================================
#         with torch.no_grad():
#             f.weight = nn.Parameter(torch.tensor(Weights,dtype=torch.float32))
#             f.bias = nn.Parameter(torch.tensor(Biases,dtype=torch.float32))
# =============================================================================
