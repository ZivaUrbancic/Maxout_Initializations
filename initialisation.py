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
    weight_and_bias : ...

    Returns
    -------
    Function R^n -> R
        Affine linear function.

    '''
    def f(x):
        return np.dot(x,weights_and_bias[:-1]) + weights_and_bias[-1]
    return f



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
    differences = Y - np.mean(Y, axis = 0)
    return np.sum(differences**2)/Y.shape[0]



def L2_region_cost(indices, Y):
    region_labels = np.array(Y[indices])
    return len(indices)*vector_variance(region_labels)



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
        projections = calculate_projections(w, data)     # calculate proj. of pts onto weight
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
    return costs[len(costs) - k - 1] > 0



# model: Maxout network as in relu.py
# X: training data
# Y: training targets
def reinitialise_ReLU_network(model, X, Y):
    # list of layers
    Layers = [layer for layer in model.children()]
    N = X.shape[0]
    R = initialise_region_matrix(N)
    C = initialise_costs_vector(N)
    reinitialise_unit = True

    stage = 1 # stages of the reinitialisation process (see below)
    for l, layer in enumerate(Layers):
        # matrix representing the weights and biases of a unit:
        WB = []
        for k in range(layer.out_features):

            # stage 0 (does not occur in ReLU networks, only for maxout networks):
            # not enough regions for running special reinitialisation routines
            # keep existing parameters until enough regions are instantiated

            # stage 1:
            # use special reinitialisation routines until regions fall below certain size
            if stage == 1:
                print("reinitialising layer ", l," unit ", k)
                wb = hyperplanes_through_largest_regions(X, R, C)
                R, C = update_regions_and_costs(R, C,
                                                [linear(wbj) for wbj in wb],
                                                X, Y, L2_region_cost)
                WB += [wb[1]] # wb[0] contains only zeroes
                if not largest_regions_have_positive_cost(C,0):
                    stage = 2

            # stage 2:
            # keep existing parameters until the end of the reinitialisation
            else:
                print("keeping layer ", l, " unit ", k)
                w = [layer.weight[k,:].detach().numpy()]
                b = [layer.bias[k].detach().numpy()]
                WB += [np.append(w, b)]

        WB = np.array(WB) # converting to np.array to improve performance of next steps
        Weights = WB[:,:-1]
        Biases = WB[:,-1]

        # compute image of the dataset under the current parameters:
        Weights = torch.tensor(Weights, dtype = torch.float32)
        Biases = torch.tensor(Biases, dtype = torch.float32)
        layer.weight = nn.Parameter(Weights)
        layer.bias = nn.Parameter(Biases)
        with torch.no_grad():
            Xtemp = np.array(layer(torch.tensor(X)))

        # adjust weights and biases to control the variance:
        Weights, Biases = fix_variance(Xtemp, Weights, Biases)
        layer.weight = nn.Parameter(Weights)
        layer.bias = nn.Parameter(Biases)

        # compute image of the dataset under the adjusted parameters
        with torch.no_grad():
            X = np.array(layer(torch.tensor(X)))

    return C


# model: Maxout network as in maxout.py
# X: training data
# Y: training targets
def reinitialise_Maxout_network(model, X, Y):
    # list of sublayers, layer i = sublayers i, i+1, ..., i+maxout_rank-1
    Sublayers = [sublayer for sublayer in model.children()]
    maxout_rank = model.maxout_rank
    N = X.shape[0] # number of data points
    R = initialise_region_matrix(N)
    C = initialise_costs_vector(N)

    stage = 0 # stages of the reinitialisation process (see below)
    for l in range(0, len(Sublayers), maxout_rank):
        # list of matrices, each matrix represents the weights and biases of a sublayer:
        WB = [[] for j in range(maxout_rank)]
        Layer = [Sublayers[l+p] for p in range(maxout_rank)]

        for k in range(Layer[0].out_features):

            # stage 0:
            # not enough regions for running special reinitialisation routines
            # keep existing parameters until enough regions are instantiated
            if stage == 0:
                print("keeping layer ", l//maxout_rank, " unit ", k)
                w = [sublayer.weight[k,:].detach().numpy() for sublayer in Layer]
                b = [sublayer.bias[k].detach().numpy() for sublayer in Layer]
                wb = [np.concatenate((w[j], [b[j]])) for j in range(len(w))]
                R, C = update_regions_and_costs(R, C,
                                                [linear(wbj) for wbj in wb],
                                                X, Y, L2_region_cost)
                for j, WBj in enumerate(WB):
                    WBj += [np.append(w[j], [b[j]])]
                if largest_regions_have_positive_cost(C, maxout_rank - 2): # returns false if number of regions < maxout_rank - 2
                    stage = 1

            # stage 1:
            # use special reinitialisation routines until regions fall below certain size
            elif stage == 1:
                print("reinitialising layer ", l//maxout_rank," unit ", k)
                wb = hyperplanes_through_largest_regions(X, R, C,
                                                         maxout = maxout_rank)
                R, C = update_regions_and_costs(R, C,
                                                [linear(wbj) for wbj in wb],
                                                X, Y, L2_region_cost)
                for j, WBj in enumerate(WB):
                    WBj += [wb[j]]
                if not largest_regions_have_positive_cost(C, maxout_rank - 2):
                    stage = 2

            # stage 2:
            # keep existing parameters until the end of the reinitialisation
            else:
                print("keeping layer ", l//maxout_rank, " unit ", k)
                w = [sublayer.weight[k,:].detach().numpy() for sublayer in Layer]
                b = [sublayer.bias[k].detach().numpy() for sublayer in Layer]
                for j, WBj in enumerate(WB):
                    WBj += [np.append(w[j], [b[j]])]

        WB = np.array(WB) # converting to np.array to improve performance of next steps

        # compute image of the dataset under the current parameters:
        for j in range(maxout_rank):
            Weights = torch.tensor(WB[j,:,:-1], dtype = torch.float32)
            Biases = torch.tensor(WB[j,:,-1], dtype = torch.float32)
            Layer[j].weight = nn.Parameter(Weights)
            Layer[j].bias = nn.Parameter(Biases)
        with torch.no_grad():
            sublayer_images = [np.array(sublayer(torch.tensor(X)))
                           for sublayer in Layer]
            sublayer_images = np.array(sublayer_images)
            Xtemp = np.amax(sublayer_images, axis = 0)

        # adjust weights and biases to control the variance:
        for j in range(maxout_rank):
            Weights = torch.tensor(WB[j,:,:-1], dtype = torch.float32)
            Biases = torch.tensor(WB[j,:,-1], dtype = torch.float32)
            Weights, Biases = fix_variance(Xtemp, Weights, Biases)
            Layer[j].weight = nn.Parameter(Weights)
            Layer[j].bias = nn.Parameter(Biases)

        # compute image of the dataset under the adjusted parameters
        with torch.no_grad():
            sublayer_images = [np.array(sublayer(torch.tensor(X)))
                           for sublayer in Layer]
            sublayer_images = np.array(sublayer_images)
            X = np.amax(sublayer_images, axis = 0)


    return C





###
# Unused code:
###
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
