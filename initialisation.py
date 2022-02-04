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
        vals = np.zeros(len(functions)); # making vals np.array to speed up np.argmax
        for i,f in enumerate(functions):
            vals[i] = f(x)
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

    sorted_X = X[R[:,-1]]
    regs = np.array(regions(sorted_X, functions)).reshape(-1,1)

    #print(r[:,-1])
    indices = R[:, -1].reshape(-1,1)

    before_regions = regions_from_costs(C)

    R = np.concatenate((R[:,:-1],
                        regs,
                        indices),
                       axis=1)
    sorted_row_indices = np.lexsort(np.rot90(R))
    R = R[sorted_row_indices]
    C = C[sorted_row_indices]

    after_regions = regions_from_matrix(R)
    #print(after_regions)

    i = 0
    for region in after_regions:

        if not region == before_regions[i]:
            if region[1] == -1:
                data_indices = R[region[0]:, -1]
            else:
                data_indices = R[region[0] : region[1], -1]
            #print(data_indices)
            C[region[0]] = region_cost(data_indices, Y)

        if region[1] == before_regions[i][1]:
            i += 1

    return R, C

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
    #print(costs)

    sorted_region_indices = np.argsort(costs)[::-1]
    #print(sorted_region_indices)

    k = min(X.shape[1], len(sorted_region_indices))

    medians = []

    for i in range(k):
        matrix_indices = regions[sorted_region_indices[i]]
        if matrix_indices[0] == R.shape[0] - 1:
            data_indices = [R[-1, -1]]
        else:
            data_indices = R[matrix_indices[0] : matrix_indices[1], -1]
        data = X[data_indices]
        if marginal:
            medians += [marginal_median(data)]
        else:
            medians += [geometric_median(data)]

    return hyperplane_through_points(medians)



def hyperplanes_through_largest_regions(X, R, C,
                                      maxout = None):

    if maxout == None:
        rank = 2
    else:
        rank = maxout

    regions = regions_from_costs(C)
    costs = C[[pair[0] for pair in regions]]
    #print(costs)

    sorted_region_indices = np.argsort(costs)[::-1]

    #k = min(X.shape[1], len(sorted_region_indices))

    #medians = []
    w = np.random.normal(size = X.shape[1])
    splits = []

    for i in range(rank - 1):
        matrix_indices = regions[sorted_region_indices[i]]
        if matrix_indices[0] == R.shape[0] - 1:
            data_indices = [R[-1, -1]]
        else:
            data_indices = R[matrix_indices[0] : matrix_indices[1], -1]
        data = X[data_indices]
        #medians += [geometric_median(data)]
        projections = calculate_projections(w, data)                  # calculate proj. of pts onto weight
        projections = np.sort(projections)
        splits += [compute_splits(projections, 2)]       # calculate splits between batches

    factors, biases = compute_factors_and_biases(splits, maxout)
    factors = factors.reshape(len(factors), 1)
    w = w.reshape(1,len(w))
    W = np.matmul(factors, w)

    return np.concatenate((W, biases.reshape(len(biases), 1)), axis=1)

def compute_factors_and_biases(splits, maxout):
    biases = np.zeros(len(splits)+1)
    if maxout is None:
        factors = np.array([0,1])
        biases[1] = -splits[0]
    else:
        factors = np.linspace(-1, 1, len(splits)+1)
        for i in range(len(factors)-1):
            biases[i+1] = biases[i] + splits[i]*(factors[i]-factors[i+1])
    return factors, biases

def compute_splits(projections, maxout):
    # print(len(projections))
    if maxout is None:
        k = 2
    else:
        k = maxout
    points_per_region = np.round(len(projections)/k).astype(int)
    #print("Points per region: ", points_per_region)
    splits = np.zeros(k-1)
    for i in range (k-1):
        splits[i]=1/2*(projections[(i+1)*points_per_region - 1] + projections[(i+1)*points_per_region])
    return splits

def calculate_projections(w, data):
    data_size, b = data.shape
    c = 0
    proj = np.zeros(data_size)
    for x in data:
        #print("x.type: ", type(x), ", w.type: ", type(w))
        proj[c] = np.dot(x, w)
        c= c+1
    return proj


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

def maxout_activation(weights_and_biases):
    functions = []
    for i in range(weights_and_biases.shape[0]):
        functions += [linear(weights_and_biases[i])]
    return functions



def fix_variance(X, weights, biases):
    Xvar = np.var(X, axis=0)
    scale_factor = np.reciprocal(np.sqrt(Xvar))
    weights = weights * scale_factor.reshape(len(scale_factor), 1) #np.matmul(np.diag(scale_factor), weights)
    biases = np.multiply(scale_factor, biases)
    return weights, biases




def largest_regions_have_positive_cost(C, k):

    costs = np.sort([c for c in C if c > -1])
    # print("region cost:", costs[len(costs)-k-1])
    return costs[len(costs) - k - 1] > 0




def reinitialise_ReLU_network(model, X, Y):
    Layers = [layer for layer in model.children()]
    N = X.shape[0]
    R = initialise_region_matrix(N)
    C = initialise_costs_vector(N)
    l = 0
    for layer in Layers:
        l += 1
        W=[]
        reinitialise_unit = True
        for k in range(layer.out_features):
            if reinitialise_unit:
                print("reinitialising layer ",l," unit ",k)
                w = hyperplanes_through_largest_regions(X, R, C, maxout=None)
                w = w[1]
                R, C = update_regions_and_costs(R, C, [linear(w),zero], X, Y, L2_region_cost)
                W += [w]
                reinitialise_unit = largest_regions_have_positive_cost(C, 0)#layer.in_features)
            else:
                print("keeping layer ",l," unit ",k)
                w = layer.weight[k,:]
                ###############################################################
                # Above I switched k and :
                ###############################################################
                #print("layer.weight[:,k] ", w)
                w = w.detach().numpy()
                b = layer.bias[k]
                b = b.detach().numpy()
                w = np.append(w, [b])
                W += [w]
        W = np.array(W)
        Weights = W[:,:-1]
        Biases = W[:,-1]

        # Compute the image of X:
        Weights = torch.tensor(Weights, dtype = torch.float32)
        Biases = torch.tensor(Biases, dtype = torch.float32)
        layer.weight = nn.Parameter(Weights)
        layer.bias = nn.Parameter(Biases)
        with torch.no_grad():
            Xtemp = np.array(layer(torch.tensor(X)))

        # Fix the weights and biases to prevent imploding and exploding activations:
        Weights, Biases = fix_variance(Xtemp, Weights, Biases)
        layer.weight = nn.Parameter(Weights)
        layer.bias = nn.Parameter(Biases)
        with torch.no_grad():
            X = np.array(layer(torch.tensor(X)))

        # Abort reinitialisation if necessary:
        if not reinitialise_unit:
            #print("Stopping reinitialisation due to lack of large regions.")
            break




def reinitialise_Maxout_network(model, X, Y):
    Sub_Layers = [sub_layer for sub_layer in model.children()]
    N = X.shape[0]
    R = initialise_region_matrix(N)
    C = initialise_costs_vector(N)
    maxout_rank = model.maxout_rank
    reinitialise_unit = True

    mode = 0

    for l in range(0, len(Sub_Layers), maxout_rank):

        W = [[] for j in range(maxout_rank)] # list of matrices, one per rank
        Layer = [Sub_Layers[l+p] for p in range(maxout_rank)]

        for k in range(0, Layer[0].out_features):

            if mode == 0:
                print("keeping layer ", l//maxout_rank, " unit ", k)
                w = [sub_layer.weight[k,:].detach().numpy() for sub_layer in Layer]
                b = [sub_layer.bias[k].detach().numpy() for sub_layer in Layer]
                wb = [np.concatenate((w[j], [b[j]])) for j in range(len(w))]

                R, C = update_regions_and_costs(R, C,
                                                [linear(wj) for wj in wb],
                                                X, Y, L2_region_cost)
                for j, Wj in enumerate(W):
                    Wj += [np.append(w[j], [b[j]])]

                if largest_regions_have_positive_cost(C, maxout_rank - 2):
                    mode = 1

            elif mode == 1:
                print("reinitialising layer ", l//maxout_rank," unit ", k)
                w = hyperplanes_through_largest_regions(X, R, C,
                                                        maxout = maxout_rank)
                R, C = update_regions_and_costs(R, C,
                                                [linear(wj) for wj in w],
                                                X, Y, L2_region_cost)
                for j, Wj in enumerate(W):
                    Wj += [w[j]]

                if not largest_regions_have_positive_cost(C, maxout_rank - 2):
                    mode = 2

            else:
                print("keeping layer ", l//maxout_rank, " unit ", k)
                w = [sub_layer.weight[k,:].detach().numpy() for sub_layer in Layer]
                b = [sub_layer.bias[k].detach().numpy() for sub_layer in Layer]
                for j, Wj in enumerate(W):
                    Wj += [np.append(w[j], [b[j]])]

        W = np.array(W)

        for j in range(maxout_rank):
            Weights = torch.tensor(W[j,:,:-1], dtype = torch.float32)
            Biases = torch.tensor(W[j,:,-1], dtype = torch.float32)
            Layer[j].weight = nn.Parameter(Weights)
            Layer[j].bias = nn.Parameter(Biases)

        with torch.no_grad():
            sub_layer_images = [np.array(sub_layer(torch.tensor(X)))
                           for sub_layer in Layer]
            sub_layer_images = np.array(sub_layer_images)
            Xtemp = np.amax(sub_layer_images, axis = 0)

        for j in range(maxout_rank):
            # Fix the weights and biases to prevent imploding and exploding activations:
            Weights = torch.tensor(W[j,:,:-1], dtype = torch.float32)
            Biases = torch.tensor(W[j,:,-1], dtype = torch.float32)
            Weights, Biases = fix_variance(Xtemp, Weights, Biases)
            Layer[j].weight = nn.Parameter(Weights)
            Layer[j].bias = nn.Parameter(Biases)

        with torch.no_grad():
            sub_layer_images = [np.array(sub_layer(torch.tensor(X)))
                           for sub_layer in Layer]
            sub_layer_images = np.array(sub_layer_images)
            X = np.amax(sub_layer_images, axis = 0)

        # Abort reinitialisation if necessary:
        if not reinitialise_unit:
            #print("Stopping reinitialisation due to lack of large regions.")
            break
