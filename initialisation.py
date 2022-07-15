#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 13:33:28 2021

@author: ziva
"""

import torch
import torch.nn as nn
import numpy as np
#from matplotlib import cm
from scipy.linalg import null_space
from scipy.spatial.distance import cdist, euclidean

import random

# random.seed(234234)
# np.random.seed(23423)
# torch.manual_seed(4654657)

# todo: find out how to get transformed datapoints from train_dataset and eliminate the need for train_loader
def sample_dataset(train_dataset, train_loader, N):
    '''
    Samples N points per label in the train_loader and returns
    the sampled points as a single np.array and labels as a list.
    train_loader is required to access the transformed datapoints.

    Parameters
    ----------
    train_dataset : torchvision.datasets
        Dataset.
    train_loader : torch.utils.data.DataLoader
        Loader.
    N : int
        Number of points per class to be sampled.

    Returns
    -------
    np array
        Sampled data.
    Y : list of ints
        int-valued labels.

    '''

    # Extract the normalised datapoints and normalised labels.
    XX = torch.cat([images for (images,labels) in train_loader])
    YY = torch.cat([labels for (images,labels) in train_loader])
    targets = [target for target in train_dataset.class_to_idx.values()]
    targets_counter = np.zeros(len(targets))


    X = []
    Y = []
    for x,y in zip(XX,YY):
        for i,t in enumerate(targets):
            if t == y and targets_counter[i]<N:
                # make x into a numpy.ndarray, if it is not
                if type(x)==type(torch.Tensor(1)):
                    x = x.numpy()
                X += [x]
                # convert singleton integer tensor to int:
                Y += [int(y)]
                targets_counter[i] += 1
        if len(Y) == N*len(targets):
            break

    return np.array(X).astype('float32'), Y

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
        vals = np.array([f(x) for f in functions])
        reg.append(np.argmax(vals))

    return reg

def class_to_unit_vector(class_labels):
    # where the class labels are integers
    n = int(torch.max(class_labels)) + 1
    unit_vecs = np.eye(n)
    Rn_labels = np.array([unit_vecs[int(i)] for i in class_labels])
    return Rn_labels

def L2_region_cost(indices, Y, number_of_classes):
    '''
    Compute the cost of a region with respect to an L2 error
    function such as MSE.

    If the overall cost of the network is proportional to
    sum_i L2(F(x_i), y_i)
    then the cost of a specific region is proportional to
    sum_R L2(F(x_i), y_i).
    If we assume the network is constant in a given region we can define
    the cost of a region as
    c(R) := min_b sum_R (b, y_i).

    Parameters
    ----------
    indices : list of ints
        The indices of the points in the region.
    Y : list of ints
        Total list of labels for all data.
    number_of_classes : int
        The number of classes from which the labels are drawn.

    Returns
    -------
    float
        The cost of the region.

    '''

    region_labels = np.zeros((number_of_classes, len(indices)))
    for i, y in enumerate(Y):
        region_labels[i, y] = 1
    differences = region_labels - np.mean(region_labels, axis = 0)
    return np.sum(differences**2)

def CE_region_cost(indices, Y, number_of_classes):
    '''
    Compute the cost of a region with respect to cross entropy
    loss.

    If the overall cost of the network is proportional to
    sum_i CE(F(x_i), y_i)
    then the cost of a specific region is proportional to
    sum_R CE(F(x_i), y_i).
    If we assume the network is constant in a given region we can define
    the cost of a region as
    c(R) := min_b sum_R (b, y_i).

    Parameters
    ----------
    indices : list of ints
        The indices of the points in the region.
    Y : list of ints
        Total list of labels for all data.
    number_of_classes : int
        The number of classes from which the labels are drawn.

    Returns
    -------
    float
        The cost of the region.

    '''

    region_labels = torch.tensor([Y[i] for i in indices])
    n = len(region_labels)
    mean = [0]*number_of_classes
    for k in region_labels:
        mean[int(k)] += 1/n
    mean_dist = torch.log(torch.tensor([mean]*n))
    CE = nn.CrossEntropyLoss(reduction = 'sum')(mean_dist, region_labels)
    return CE

def regions_from_costs(costs):
    '''
    Read cost vector and identify the start points of the regions. Return a
    list of pairs [s_i, s_{i+1}] of start points of consecutive regions.

    Parameters
    ----------
    costs : np array
        Cost vector.

    Returns
    -------
    list
        List of pairs of start indices of consecutive regions.

    '''

    region_start_points = []
    for n, c in enumerate(costs):
        if c >= 0:
            region_start_points += [n]
    region_start_points += [-1]

    return [[region_start_points[i],
             region_start_points[i+1]]
            for i in range(len(region_start_points) - 1)]

def regions_from_matrix(R):
    '''
    Read regions matrix and identify the start points of the regions. Return
    a list of pairs [s_i, s_{i+1}] of start points of consecutive regions.

    Parameters
    ----------
    R : np array
        Regions matrix.

    Returns
    -------
    list
        List of pairs of start indices of consecutive regions.

    '''

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

def update_regions_and_costs(R, C, functions, X, Y,
                             region_cost, number_of_classes):
    '''
    Update the regions matrix and cost vector. Take a list of functions of
    length maxout_rank (or 2 in the ReLU case where the first is the zero
    function) and decide which function is maximal at each point x. This gives
    a vector of indices which is appended as the penultimate column of the
    region matrix. We also recompute the costs for the regions and update the
    cost vector to include these.

    Parameters
    ----------
    R : np array
        Region matrix.
    C : np array
        Cost vector.
    functions : list
        List of functions R^n -> R.
    X : np array
        Data set N x n.
    Y : np array
        Data labels.
    region_cost : function
        Region cost function.
    number_of_classes : int
        The number of classes from which Y is drawn, in the case of multi-class
        classification.

    Returns
    -------
    R : np array
        Updated regions matrix.
    C : np array
        Updated cost vector.

    '''
    # Sort X according to the order given by the indices in the region matrix,
    # to associate each data point to its region. Compute the index of the
    # function which is maximal at x.
    indices = R[:,-1]
    sorted_X = X[indices]
    regs = np.array(regions(sorted_X, functions)).reshape(-1,1)
    indices = indices.reshape(-1,1)

    # Compute the region indices from the cost vector, and append the vector
    # of region indices for the new unit.
    before_regions = regions_from_costs(C)
    R = np.concatenate((R[:,:-1],
                        regs,
                        indices),
                       axis=1)

    # Sort the matrix and costs to collect the regions together.
    sorted_row_indices = np.lexsort(np.rot90(R))
    R = R[sorted_row_indices]
    C = C[sorted_row_indices]

    # Identify the newly created regions (that contain data) and compute the
    # new costs for all changed regions.
    after_regions = regions_from_matrix(R)
    i = 0
    for region in after_regions:

        if not region == before_regions[i]:
            if region[1] == -1:
                data_indices = R[region[0]:, -1]
            else:
                data_indices = R[region[0] : region[1], -1]
            C[region[0]] = region_cost(data_indices, Y, number_of_classes)

        if region[1] == before_regions[i][1]:
            i += 1

    return R, C

def hyperplanes_through_largest_regions(X, R, C,
                                        maxout = None, w = None):
    '''
    Identify the regions with largest cost from the cost vector
    and splits that region with #maxout hyperplanes.

    Parameters
    ----------
    X : np array
        Data matrix.
    R : np array
        Regions matrix.
    C : np array
        Costs vector.
    maxout : int, optional
        Maxout rank. The default is None.
    w : np array, optional
        Weight vector to be used, if it needs to be specified.
        The default is None.

    Returns
    -------
    np array
        Weight and biases vector, interpreted as a weight in R^{n+1}.

    '''

    # Deal with the ReLU case as a maxout rank 2.
    if maxout == None:
        rank = 2
    else:
        rank = maxout

    # Find the indices of the existing regions from the costs vector,
    # and collect their costs in a new vector. Sort this by size.
    regions = regions_from_costs(C)
    costs = C[[pair[0] for pair in regions]]
    sorted_region_indices = np.argsort(costs)[::-1]

    # Randomly initialise the weight in the (usual) case that it is
    # not already specified.
    if w is None:
        w = np.random.normal(size = X.shape[1])

    # Compute the splitting points for the largest-cost regions, along
    # the axis of the new weight w.
    splits = []
    for i in range(rank - 1):

        # Pick out the i^th largest-cost region, and find the indices
        # of the regions matrix that correspond to that region.
        matrix_indices = regions[sorted_region_indices[i]]

        # Find the indices of the data matrix X that correspond to
        # points in this region.
        if matrix_indices[0] == R.shape[0] - 1:
            data_indices = [R[-1, -1]]
        else:
            data_indices = R[matrix_indices[0] : matrix_indices[1], -1]

        # Slice out the data in the region, using these indices.
        data = X[data_indices]

        # Project the data onto the axis specified by the weight
        # vector, and compute the bias which will split it into
        # two new regions.
        projections = np.dot(data, w)
        projections.sort()
        splits += [compute_splits(projections, 2)]

    # Compute factors to rescale the weight vector w for each unit,
    # and biases for each, such that the decision boundary occurs
    # at the splitting points we calculated.
    factors, biases = compute_factors_and_biases(splits, maxout)
    W = np.array([f*w for f in factors])

    return np.concatenate((W, biases.reshape(-1, 1)), axis=1)


def compute_factors_and_biases(splits, maxout = None):
    '''
    Compute a factor and bias for each maxout unit such that the
    decision boundary occurs at the splitting points of splits.

    Parameters
    ----------
    splits : np array
        Splitting points.
    maxout : int, optional
        Rank. The default is None.

    Returns
    -------
    factors : np array
        Factors.
    biases : np array
        Biases.

    '''
    biases = np.zeros(len(splits)+1)
    if maxout is None:
        factors = np.array([0,1])
        biases[1] = -splits[0]
    else:
        factors = np.linspace(-1, 1, len(splits)+1)
        for i in range(len(factors)-1):
            biases[i+1] = biases[i] + splits[i]*(factors[i]-factors[i+1])
    return factors, biases

def compute_splits(projections, number_of_regions = 2):
    '''
    Compute the midpoint of #number_of_regions equally large regions.

    Parameters
    ----------
    projections : np array
        1D vector of data projected onto the weight vector w.
    number_of_regions : int, optional
        The number of regions into which the data will be split.
        The default is 2.

    Returns
    -------
    splits : np array
        Splitting points.

    '''

    points_per_region = len(projections) // number_of_regions
    splits = [(projections[(i+1)*points_per_region - 1]
                   + projections[(i+1)*points_per_region])/2
              for i in range (number_of_regions-1)]
    return np.array(splits)

# def fix_variance(X, weights, biases):
#     Xvar = np.var(X, axis=0)
#     scale_factor = np.reciprocal(np.sqrt(Xvar))
#     weights = weights * scale_factor.reshape(len(scale_factor), 1) #np.matmul(np.diag(scale_factor), weights)
#     biases = np.multiply(scale_factor, biases)
#     return weights, biases

def fix_child_variance(child, X):
    '''
    Rescale the weights and biases of a layer 'child' to normalise the
    image X of the data. If the layer is linear we rescale componentwise, and so will take the
    standard deviation over each component of X. If the layer is
    convolutional then, when we interpret the action of the kernel as a
    matrix, each row is a permutation of zeros and the kernel weights. There
    is just one parameter to rescale each row, but all rows must have the
    same weights, so there is only one free parameter to rescale the entire
    kernel.

    Update the parameters of child in-place.

    Parameters
    ----------
    child : nn.Linear
        Torch layer.
    X : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    # Choose the axes of the data along which to compute the standard
    # deviation, depending on the type of the layer.
    if type(child) == torch.nn.modules.conv.Conv2d:
        conv = 2
        ax = (0,2,3)
    elif type(child) == torch.nn.modules.conv.Conv1d:
        conv = 1
        ax = (0,2)
    else:
        conv = 0
        ax = 0

    # Rescale weights and biases according to the type of layer.
    std_scale = torch.tensor(np.std(X, axis = ax))
    std_scale[std_scale == 0] = 1
    with torch.no_grad():
        if conv == 2:
            child.weight /= std_scale.reshape(-1, 1, 1, 1)
        elif conv == 1:
            child.weight /= std_scale.reshape(-1, 1, 1)
        else:
            child.weight /= std_scale.reshape(-1, 1)
        child.bias /= std_scale

def k_th_largest_region_cost(C,k):
    '''
    Compute the cost of the k^th largest region (where k starts at 0),
    and returns -1 if k > #linear regions.

    Parameters
    ----------
    C : np array
        Cost vector.
    k : int
        k.

    Returns
    -------
    float
        Cost of the k^th largest region.

    '''

    # If C is empty we will jump straight to stage 2 in the reinitialisation,
    # which we signal by returning 0 here.
    if len(C) == 0:
        return 0
    C1 = C.copy()
    C1[::-1].sort()
    return C1[k]


# def marginal_median(Y):
#     '''
#     Find the marginal median of a data set Y.

#     Parameters
#     ----------
#     Y : List or Array
#         Data set in R^n.

#     Returns
#     -------
#     Array
#         Component-wise median point in R^n.

#     '''

#     d= Y.shape[1]
#     point = []

#     for n in range(d):
#         point += [np.median(Y[:,n])]

#     return np.array(point)



# # copied from https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points
# def geometric_median(X, eps=1e-5):

#     y = np.mean(X, 0)

#     while True:
#         D = cdist(X, [y])
#         nonzeros = (D != 0)[:, 0]

#         Dinv = 1 / D[nonzeros]
#         Dinvs = np.sum(Dinv)
#         W = Dinv / Dinvs
#         T = np.sum(W * X[nonzeros], 0)

#         num_zeros = len(X) - np.sum(nonzeros)
#         if num_zeros == 0:
#             y1 = T
#         elif num_zeros == len(X):
#             return y
#         else:
#             R = (T - y) * Dinvs
#             r = np.linalg.norm(R)
#             rinv = 0 if r == 0 else num_zeros/r
#             y1 = max(0, 1-rinv)*T + min(1, rinv)*y

#         if euclidean(y, y1) < eps:
#             return y1
#         y = y1

def reinitialise_network(model, X, Y, return_cost_vector = False, adjust_regions = True, adjust_variance = True):
    N = X.shape[0] # number of data points
    R = initialise_region_matrix(N)
    C = initialise_costs_vector(N)

    for l, child in enumerate(model.children()):

        if type(child)==torch.nn.modules.conv.Conv1d or type(child)==torch.nn.modules.conv.Conv2d:
            print("Reinitialising layer", l,"of type Conv1d or Conv2d")
            X, R, C = reinitialise_conv_layer(child, X, Y, R, C,
                                              # return_cost_vector = return_cost_vector, # todo?
                                              adjust_regions = adjust_regions,
                                              adjust_variance = adjust_variance)

        else:
            # 1D-layers, flatten data if multi-dimensional, e.g., 2D images in MNIST and CIFAR10
            if len(X.shape)>2:
                X = np.array([x.flatten() for x in X])

            if type(child)==torch.nn.modules.linear.Linear:
                print("Reinitialising layer", l,"of type ReLU")
                X, R, C = reinitialise_relu_layer(child, X, Y, R, C,
                                              # return_cost_vector = return_cost_vector, # todo?
                                              adjust_regions = adjust_regions,
                                              adjust_variance = adjust_variance)

            elif type(child)==torch.nn.modules.container.ModuleList:
                print("Reinitialising layer", l,"of type Maxout")
                X, R, C = reinitialise_maxout_layer(child, X, Y, R, C,
                                                    return_cost_vector = return_cost_vector,
                                                    adjust_regions = adjust_regions,
                                                    adjust_variance = adjust_variance)

            else:
                print("Ignoring child of type", type(child))
                # todo: check if layer supported, print warning if not

    return C

def reinitialise_maxout_layer(children, X, Y, R, C, return_cost_vector = False, adjust_regions = True, adjust_variance = True):
    if type(R) == bool or type(C) == bool:
        N = X.shape[0] # number of data points
        assert R == False and C == False
        R = initialise_region_matrix(N)
        C = initialise_costs_vector(N)

    maxout_rank = len(children)
    number_of_classes=max(Y)+1

    # step 0: check whether maxout_rank > number of regions
    c = k_th_largest_region_cost(C, maxout_rank-2)

    if not adjust_regions:
        # no adjusting regions, go to stage 0 or 2 depending on whether
        # cost vector needs to be computed
        if return_cost_vector:
            stage = 0
        else:
            stage = 2
    else:
        if return_cost_vector:
            # adjusting regions and cost vector needs to be computed,
            # go to stage 1 if k_th_largest_region has positive cost (= region adjustments necessary)
            # go to stage 0 otherwise
            if c > 0:
                stage = 1
            else:
                stage = 0
        else:
            # adjusting regions and cost vector needs to be computed,
            # go to stage 1 if k_th_largest_region has positive cost (= region adjustments necessary)
            # go to stage 2 if k_th_largest_region has 0 cost (= region adjustments done)
            # go to stage 0 if k_th_largest_region has negative cost (more regions required before adjustments)
            if c > 0:
                stage = 1
            if c == -1:
                stage = 0
            if c == 0:
                stage = 2


    # step 1: reintialise parameters
    for k in range(children[0].out_features):
        # stage 0:
        # not enough regions for running special reinitialisation routines
        # keep existing parameters until enough regions are instantiated
        if stage == 0:
            print("keeping unit", k)
            w = [child.weight[k,:].detach().numpy() for child in children]
            b = [child.bias[k].detach().numpy() for child in children]
            wb = [np.concatenate((w[j], [b[j]])) for j in range(len(w))]
            R, C = update_regions_and_costs(R, C,
                                            [linear(wbj) for wbj in wb],
                                            X, Y, CE_region_cost,
                                            number_of_classes)

            k_cost = k_th_largest_region_cost(C, maxout_rank - 2)
            if adjust_regions and k_cost > 0: # returns false if number of regions < maxout_rank - 2
                stage = 1
            if not return_cost_vector and k_cost == 0: # returns false if number of regions < maxout_rank - 2
                stage = 2

        # stage 1:
        # use special reinitialisation routines until regions fall below certain size
        elif stage == 1:
            print("reinitialising unit", k)
            wb = hyperplanes_through_largest_regions(X, R, C,
                                                     maxout = maxout_rank)


            R, C = update_regions_and_costs(R, C,
                                            [linear(wbj) for wbj in wb],
                                            X, Y, CE_region_cost,
                                            number_of_classes)
            with torch.no_grad():
                for j, child in enumerate(children):
                    child.weight[k, :] = nn.Parameter(torch.tensor(wb[j][:-1]))
                    child.bias[k] = nn.Parameter(torch.tensor(wb[j][-1]))

            if k_th_largest_region_cost(C, maxout_rank - 2) == 0:
                if not return_cost_vector:
                    stage = 2
                else:
                    stage = 0

        # stage 2:
        # all regions have cost 0, keep remaining parameters, no need to compute cost vector
        elif stage == 2:
            print("keeping unit",k,"onwards")
            R = []
            C = []
            break


    # step 3: adjust weights to control variance and forward X
    # compute image of the dataset under the current parameters:
    if adjust_variance == True:
        with torch.no_grad():
            Xtemp = np.amax([child(torch.tensor(X)).numpy() for child in children],
                            axis = 0)

        # adjust weights and biases to control the variance:
        for child in children:
            fix_child_variance(child, Xtemp)

    # compute image of the dataset under the adjusted parameters
    with torch.no_grad():
        X = np.amax([child(torch.tensor(X)).numpy() for child in children],
                        axis = 0)

    return X, R, C


def reinitialise_relu_layer(child, X, Y, R = False, C = False, adjust_regions = True, adjust_variance = True):

    if type(R) == bool or type(C) == bool:
        N = X.shape[0] # number of data points
        assert R == False and C == False
        R = initialise_region_matrix(N)
        C = initialise_costs_vector(N)

    number_of_classes = len(set(Y))
    # step 0: check whether maxout_rank > number of regions
    if k_th_largest_region_cost(C, 0) == 0 or adjust_regions == False:
        stage = 2
    elif k_th_largest_region_cost(C, 0) == -1:
        stage = 0
    else:
        stage = 1

    # step 1: reintialise parameters
    for k in range(child.out_features):
        # stage 0:
        # not enough regions for running special reinitialisation routines
        # keep existing parameters until enough regions are instantiated
        if stage == 0:
            print("keeping unit",k)
            w = child.weight[k,:].detach().numpy()
            b = child.bias[k].detach().numpy()
            wb = np.concatenate((w[j], [b[j]]))
            R, C = update_regions_and_costs(R, C,
                                            [linear(wb),zero],
                                            X, Y, CE_region_cost,
                                            number_of_classes)

            k_cost = k_th_largest_region_cost(C, 0)
            if k_cost > 0: # returns false if number of regions < maxout_rank - 2
                stage = 1
            if k_cost == 0: # returns false if number of regions < maxout_rank - 2
                stage = 2

        # stage 1:
        # use special reinitialisation routines until regions fall below certain size
        elif stage == 1:
            print("reinitialising unit",k)
            wb = hyperplanes_through_largest_regions(X, R, C)
            R, C = update_regions_and_costs(R, C,
                                            [linear(wbj) for wbj in wb],
                                            X, Y, CE_region_cost,
                                            number_of_classes)
            with torch.no_grad():
                child.weight[k, :] = nn.Parameter(torch.tensor(wb[1][:-1])) # wb[0] contains only 0s
                child.bias[k] = nn.Parameter(torch.tensor(wb[1][-1]))

            if k_th_largest_region_cost(C, 0) == 0:
                stage = 2

        # stage 2:
        # all regions have cost 0, keep remaining parameters
        elif stage == 2:
            print("keeping unit",k,"onwards")
            R = []
            C = []
            break


    # step 3: adjust weights to control variance and forward X
    # compute image of the dataset under the current parameters:
    if adjust_variance == True:
        with torch.no_grad():
            Xtemp = nn.ReLU()(child(torch.tensor(X))).numpy()

        # adjust weights and biases to control the variance:
        fix_child_variance(child, Xtemp)

    # compute image of the dataset under the adjusted parameters
    with torch.no_grad():
        X = nn.ReLU()(child(torch.tensor(X))).numpy()

    return X, R, C


child = nn.Conv2d(3, 6, (2,3), stride=1)

assert child.bias != None



def reinitialise_conv_layer(child, X, Y, R = False, C = False,
                              adjust_regions = True,
                              adjust_variance = True):

    if type(R) == bool or type(C) == bool:
        N = X.shape[0] # number of data points
        assert R == False and C == False
        R = initialise_region_matrix(N)
        C = initialise_costs_vector(N)

    if adjust_regions == False and adjust_variance == False:
        return X, R, C

    if type(X) == np.ndarray:
        X = torch.from_numpy(X.astype('float32'))

    if type(child) == torch.nn.modules.conv.Conv2d:
        child1 = nn.Conv2d(child.in_channels,
                       child.out_channels,
                       child.kernel_size,
                       child.stride,
                       child.padding,
                       child.dilation,
                       child.groups,
                       False,
                       child.padding_mode)
        X1 = child1(X).detach().numpy()
        X2 = X1.mean(axis = (-2,-1))

    elif type(child) == torch.nn.modules.conv.Conv1d:
        child1 = nn.Conv1d(child.in_channels,
                       child.out_channels,
                       child.kernel_size,
                       child.stride,
                       child.padding,
                       child.dilation,
                       child.groups,
                       False,
                       child.padding_mode)
        X1 = child1(X).detach().numpy()
        X2 = X1.mean(axis = -1)

    else:
        raise TypeError('Child must be nn.Conv1d or nn.Conv2d')

    number_of_classes = len(set(Y))
    # step 0: check whether maxout_rank > number of regions
    if k_th_largest_region_cost(C, 0) == 0 or adjust_regions == False:
        stage = 2
    else:
        stage = 1

    unit_vecs = np.eye(child.out_channels)

    # step 1: reintialise parameters
    for k in range(child.out_channels):
        # stage 1:
        # use special reinitialisation routines until regions fall
        # below certain size
        if stage == 1:
            print("reinitialising channel", k)
            wb = hyperplanes_through_largest_regions(X2, R, C, w = unit_vecs[k])

            R, C = update_regions_and_costs(R, C,
                                            [linear(wbj) for wbj in wb],
                                            X2, Y, CE_region_cost,
                                            number_of_classes)

            with torch.no_grad():
                child.bias[k] = nn.Parameter(torch.tensor(wb[1][-1]))

            if k_th_largest_region_cost(C, 0) == 0:
                stage = 2

        # stage 2:
        # all regions have cost 0, keep remaining parameters
        elif stage == 2:
            print("keeping channel",k,"onwards")
            R = []
            C = []
            break

    # step 2: adjust weights to control variance and the image of X

    # compute image of the dataset under the current parameters:
    if adjust_variance == True:
        with torch.no_grad():
            Xtemp = nn.ReLU()(child(X)).numpy()

        # adjust weights and biases to control the variance:
        fix_child_variance(child, Xtemp)

    # compute image of the dataset under the adjusted parameters
    with torch.no_grad():
        X = nn.ReLU()(child(X)).numpy()

    return X, R, C


# X = torch.zeros((20,3,10,10))
# for d in range(20):
#     for c in range(3):
#         for y in range(10):
#             for x in range(10):
#                 X[d,c,y,x] = 1000*(d+1) + 100*c + 10*y +x
# Y = np.random.randint(0,5,20)

# reinitialise_conv_layer(child, X, Y);




# def reinitialise_conv2d_layer(child, X, Y, R = False, C = False,
#                               adjust_regions = True,
#                               adjust_variance = True):

#     if type(R) == bool or type(C) == bool:
#         N = X.shape[0] # number of data points
#         assert R == False and C == False
#         R = initialise_region_matrix(N)
#         C = initialise_costs_vector(N)

#     if adjust_regions == False and adjust_variance == False:
#         return X, R, C

#     child1 = nn.Conv2d(child.in_channels,
#                        child.out_channels,
#                        child.kernel_size,
#                        child.stride,
#                        child.padding,
#                        child.dilation,
#                        child.groups,
#                        False,
#                        child.padding_mode)

#     if type(X) == np.ndarray:
#         X = torch.from_numpy(X)

#     X1 = child1(X).detach().numpy()
#     X2 = X1.mean(axis = (-2,-1))

#     number_of_classes = len(set(Y))
#     # step 0: check whether maxout_rank > number of regions
#     if k_th_largest_region_cost(C, 0) == 0 or adjust_regions == False:
#         stage = 2
#     else:
#         stage = 1

#     unit_vecs = np.eye(child.out_channels)

#     # step 1: reintialise parameters
#     for k in range(child.out_channels):
#         # stage 1:
#         # use special reinitialisation routines until regions fall
#         # below certain size
#         if stage == 1:
#             print("reinitialising channel", k)
#             wb = hyperplanes_through_largest_regions(X2, R, C, w = unit_vecs[k])

#             R, C = update_regions_and_costs(R, C,
#                                             [linear(wbj) for wbj in wb],
#                                             X2, Y, CE_region_cost,
#                                             number_of_classes)

#             with torch.no_grad():
#                 child.bias[k] = nn.Parameter(torch.tensor(wb[1][-1]))

#             if k_th_largest_region_cost(C, 0) == 0:
#                 stage = 2

#         # stage 2:
#         # all regions have cost 0, keep remaining parameters
#         elif stage == 2:
#             print("keeping channel ", k, " onwards")
#             R = []
#             C = []
#             break

#     # # Crop the image to a space of dimension c0 = width * height of kernel
#     # h, w = child.kernel_size
#     # c0 = w*h

#     # # Use a new convolutional layer which copies the hyperparameters of
#     # # child but with weights that project the image onto the i,j th component
#     # crop_conv = nn.Conv2d(child.in_channels,
#     #                       c0 * child.in_channels,
#     #                       child.kernel_size,
#     #                       stride = child.stride,
#     #                       padding = child.padding,
#     #                       dilation = child.dilation,
#     #                       groups = child.groups,
#     #                       bias = False,
#     #                       padding_mode = child.padding_mode)
#     # W = torch.zeros(crop_conv.weight.shape)
#     # for channel in range(child.in_channels):
#     #     for y in range(h):
#     #         for x in range(w):
#     #             W[channel*c0 + y*w + x, channel, y, x] = 1.

#     # with torch.no_grad():
#     #     X_cropped = crop_conv(torch.tensor(X)).numpy()

#     # # Find width and height of the image
#     # Width, Height = X.shape[-2:]

#     # X_ = []

#     # # Crop the images to the shape of the kernel (for each out channel)
#     # for k in range(child.in_channels):
#     #     # Crop each image for each kernel translation, and put them all in
#     #     # one big list of length |X| * (W - w + 1) * (H - h + 1)
#     #     crops = []
#     #     for i in range(Height - h + 1):
#     #         for j in range(Width - w + 1):
#     #             for point in range(X.shape[0]):
#     #                 crops.append([X_cropped[point,k*c0 : (k+1)*c0][p,i,j]
#     #                        for p in range(c0)])

#     #     X_.append(crops)

#     # # Create reshaped X and Y data for the new cropped images
#     # X_ = np.concatenate(X_, axis = 1)
#     # Y_ = np.tile(Y, (Height - h + 1)*(Width - w + 1))

#     # # Create a linear ReLU layer and reinitialise with the cropped data
#     # c0_child = nn.Linear(c0*child.in_channels, child.out_channels)
#     # c0_child.weight = nn.Parameter(child.weight.reshape(child.out_channels,
#     #                                                     c0*child.in_channels))
#     # c0_child.bias = child.bias
#     # reinitialise_relu_layer(c0_child, X_, Y_, adjust_regions = adjust_regions, adjust_variance = adjust_variance)

#     # # Reshape the reinitialised weights as a convolutional weight tensor
#     # # and use as weights for the original child
#     # reshaped_weights = c0_child.weight.reshape(child.weight.shape)
#     # child.weight = nn.Parameter(reshaped_weights)
#     # child.bias = c0_child.bias

#     # step 3: adjust weights to control variance and forward X
#     # compute image of the dataset under the current parameters:
#     if adjust_variance == True:
#         with torch.no_grad():
#             Xtemp = nn.ReLU()(child(torch.tensor(X))).numpy()

#         # adjust weights and biases to control the variance:
#         fix_child_variance(child, Xtemp)

#     # compute image of the dataset under the adjusted parameters
#     with torch.no_grad():
#         X = nn.ReLU()(child(X)).numpy()

#     return X, R, C
