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



# samples N points per target in the train_loader and returns
# sampled points as a single np.array and targets as a list.
# train_loader is required to access the transformed datapoints
# todo: find out how to get transformed datapoints from train_dataset and eliminate the need for train_loader
def sample_dataset(train_dataset,train_loader,N,random=False):
    XX = torch.cat([images for (images,labels) in train_loader]) # total set of normalized datapoints
    YY = torch.cat([labels for (images,labels) in train_loader]) # targets of normalized datapoints
    targets = [target for target in train_dataset.class_to_idx.values()] # set of targets (one entry per target)
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
        vals = np.zeros(len(functions)); # making vals np.array to speed up np.argmax
        for i,f in enumerate(functions):
            vals[i] = f(x)
        reg += [np.argmax(vals)]

    return reg

def class_to_unit_vector(class_labels):
    # where the class labels are integers
    n = int(torch.max(class_labels)) + 1
    unit_vecs = np.eye(n)
    Rn_labels = np.array([unit_vecs[int(i)] for i in class_labels])
    return Rn_labels

# input:
#    indices: list of integers, indices of datapoints in region
#    Y: list of integers, set of all targets in dataset
#    number_of_classes: int
# output:
#    float: region_cost of region containing points with targets
#           Y(i) for i in indices
def L2_region_cost(indices, Y, number_of_classes):
    region_labels = np.zeros((number_of_classes, len(indices)))
    for i, y in enumerate(Y):
        region_labels[i, y] = 1
    differences = region_labels - np.mean(region_labels, axis = 0)
    return np.sum(differences**2)

def CE_region_cost(indices, Y, number_of_classes):
    region_labels = torch.tensor([Y[i] for i in indices])
    n = len(region_labels)
    mean = [0]*number_of_classes
    for k in region_labels:
        mean[int(k)] += 1/n
    mean_dist = torch.log(torch.tensor([mean]*n))
    CE = nn.CrossEntropyLoss(reduction = 'sum')(mean_dist, region_labels)
    return CE

#indices = [i for i in range(6)]
#Y = torch.tensor([0,0,0,0,0,0])
#print(CE_region_cost(indices, Y, 3))

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



def update_regions_and_costs(R, C, functions, X, Y,
                             region_cost, number_of_classes):

    sorted_X = X[R[:,-1]]
    regs = np.array(regions(sorted_X, functions)).reshape(-1,1)

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
                                        maxout = None):

    if maxout == None:
        rank = 2
    else:
        rank = maxout

    regions = regions_from_costs(C)
    costs = C[[pair[0] for pair in regions]]

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
    if maxout is None:
        k = 2
    else:
        k = maxout
    points_per_region = np.round(len(projections)/k).astype(int)
    splits = np.zeros(k-1)
    for i in range (k-1):
        splits[i]=1/2*(projections[(i+1)*points_per_region - 1] + projections[(i+1)*points_per_region])
    return splits


def calculate_projections(w, data):
    data_size, b = data.shape
    c = 0
    proj = np.zeros(data_size)
    for x in data:
        proj[c] = np.dot(x, w)
        c= c+1
    return proj


def fix_variance(X, weights, biases):
    Xvar = np.var(X, axis=0)
    scale_factor = np.reciprocal(np.sqrt(Xvar))
    weights = weights * scale_factor.reshape(len(scale_factor), 1) #np.matmul(np.diag(scale_factor), weights)
    biases = np.multiply(scale_factor, biases)
    return weights, biases


def fix_child_variance(child, X):
    std_scale = torch.tensor(np.std(X, axis=0))
    std_scale[std_scale == 0] = 1
    with torch.no_grad():
        child.weight /= std_scale.reshape(len(std_scale), 1)
        child.bias /= std_scale#.reshape(len(scale_factor), 1)

# returns the cost of the k-th largest region (starting at 0)
# returns -1 if k > #linear regions
def k_th_largest_region_cost(C,k):
    # If C is empty we jump straight to stage 2, by returning 0
    if len(C) == 0:
        return 0
    C[::-1].sort()
    return C[k]


###
# Unused code:
###
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

def reinitialise_network(model, X, Y, rescale_only = False):
    N = X.shape[0] # number of data points
    R = initialise_region_matrix(N)
    C = initialise_costs_vector(N)

    skip = 0
    l = 0
    children = [child for child in model.children()]
    for i, child in enumerate(children):
        if skip>0:
            skip -= 1

        elif type(child)==torch.nn.modules.conv.Conv1d:
            print("Reinitialising layer ", l, " of type Conv1d")
            l += 1
            X, R, C = reinitialise_conv1d_layer(child, X, Y, R, C, rescale_only)

        elif type(child)==torch.nn.modules.conv.Conv2d:
            print("Reinitialising layer ", l, " of type Conv2d")
            l += 1
            X, R, C = reinitialise_conv2d_layer(child, X, Y, R, C, rescale_only)

        elif type(child)==torch.nn.modules.linear.Linear:
            if len(X.shape)>2:
                # flatten each datapoint in case input is multidimational
                # e.g. 2D images in MNIST and CIFAR10
                X = np.array([x.flatten() for x in X])
            l += 1
            if hasattr(model, 'maxout_rank'):
                print("Reinitialising layer ", l, " of type Maxout")
                X, R, C = reinitialise_maxout_layer(children[i:i+model.maxout_rank],
                                                    X, Y, R, C, rescale_only)
                skip = model.maxout_rank-1
            else:
                print("Reinitialising layer ", l, " of type ReLU")
                X, R, C = reinitialise_relu_layer(child, X, Y, R, C, rescale_only)
        else:
            print(type(child))
            # todo: check if layer supported, print warning if not
            continue


def reinitialise_maxout_layer(children, X, Y, R, C, rescale_only = False):
    maxout_rank = len(children)
    number_of_classes=max(Y)+1
    # step 0: check whether maxout_rank > number of regions
    
    if k_th_largest_region_cost(C, maxout_rank-2) == 0 or rescale_only:
        stage = 2
    elif k_th_largest_region_cost(C, maxout_rank - 2) == -1:
        stage = 0
    else:
        stage = 1
    
    # step 1: reintialise parameters
    for k in range(children[0].out_features):
        # stage 0:
        # not enough regions for running special reinitialisation routines
        # keep existing parameters until enough regions are instantiated
        if stage == 0:
            print("keeping unit ", k)
            w = [child.weight[k,:].detach().numpy() for child in children]
            b = [child.bias[k].detach().numpy() for child in children]
            wb = [np.concatenate((w[j], [b[j]])) for j in range(len(w))]
            R, C = update_regions_and_costs(R, C,
                                            [linear(wbj) for wbj in wb],
                                            X, Y, CE_region_cost,
                                            number_of_classes)

            k_cost = k_th_largest_region_cost(C, maxout_rank - 2)
            if k_cost > 0: # returns false if number of regions < maxout_rank - 2
                stage = 1
            if k_cost == 0: # returns false if number of regions < maxout_rank - 2
                stage = 2

        # stage 1:
        # use special reinitialisation routines until regions fall below certain size
        elif stage == 1:
            print("reinitialising unit ", k)
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
                stage = 2

        # stage 2:
        # all regions have cost 0, keep remaining parameters
        elif stage == 2:
            print("keeping unit ", k, " onwards")
            R = []
            C = []
            break


    # step 3: adjust weights to control variance and forward X
    # compute image of the dataset under the current parameters:
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


def reinitialise_relu_layer(child, X, Y, R = False, C = False, rescale_only = False):
    
    N = X.shape[0] # number of data points
    
    if R == False or C == False:
        assert R == False and C == False
        R = initialise_region_matrix(N)
        C = initialise_costs_vector(N)
    
    number_of_classes = len(set(Y))
    # step 0: check whether maxout_rank > number of regions
    if k_th_largest_region_cost(C, 0) == 0 or rescale_only:
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
            print("keeping unit ", k)
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
            print("reinitialising unit ", k)
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
            print("keeping unit ", k, " onwards")
            R = []
            C = []
            break


    # step 3: adjust weights to control variance and forward X
    # compute image of the dataset under the current parameters:
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

X = torch.zeros((20,3,10,10))
for d in range(20):
    for c in range(3):
        for y in range(10):
            for x in range(10):
                X[d,c,y,x] = 1000*(d+1) + 100*c + 10*y +x
Y = np.random.randint(0,5,20)                

def reinitialise_conv2d_layer(child, X, Y, R = False, C = False,
                              rescale_only = False):
    
    if not R == False and C == False:
        print('Unsupported: R or C specified for conv2d layer')
        
    # Crop the image to a space of dimension c0 = width * height of kernel
    h, w = child.kernel_size
    c0 = w*h
    
    # Use a new convolutional layer which copies the hyperparameters of
    # child but with weights that project the image onto the i,j th component
    crop_conv = nn.Conv2d(child.in_channels,
                          c0 * child.in_channels,
                          child.kernel_size,
                          child.stride,
                          child.padding,
                          child.dilation,
                          child.groups,
                          False,
                          child.padding_mode)
    W = torch.zeros(crop_conv.weight.size())
    for channel in range(child.in_channels):
        for y in range(h):
            for x in range(w):
                W[channel*c0 + y*w + x, channel, y, x] = 1.
    
    crop_conv.weight = nn.Parameter(W)
    X_cropped = crop_conv(X).detach().numpy()
    
    # Find width and height of the image
    Width, Height = X.shape[-2:]
    
    X_ = []
    
    # Crop the images to the shape of the kernel (for each out channel)
    for k in range(child.in_channels):
        # Crop each image for each kernel translation, and put them all in
        # one big list of length |X| * (W - w + 1) * (H - h + 1)
        crops = []
        for i in range(Height - h + 1):
            for j in range(Width - w + 1):
                for point in range(X.shape[0]):
                    crops.append([X_cropped[point,k*c0 : (k+1)*c0][p,i,j]
                           for p in range(c0)])
        
        X_.append(crops)
    
    # Create reshaped X and Y data for the new cropped images
    X_ = np.concatenate(X_, axis = 1)
    Y_ = np.tile(Y, (Height - h + 1)*(Width - w + 1))
    
    # Create a linear ReLU layer and reinitialise with the cropped data
    c0_child = nn.Linear(c0*child.in_channels, child.out_channels)
    c0_child.weight = nn.Parameter(child.weight.reshape(child.out_channels,
                                                        c0*child.in_channels))
    c0_child.bias = child.bias
    reinitialise_relu_layer(c0_child, X_, Y_, rescale_only = rescale_only)
    
    # Reshape the reinitialised weights as a convolutional weight tensor
    # and use as weights for the original child
    reshaped_weights = c0_child.weight.reshape(child.weight.shape)
    child.weight = nn.Parameter(reshaped_weights)
    child.bias = c0_child.bias

    # compute image of the dataset under the adjusted parameters
    with torch.no_grad():
        X = nn.ReLU()(child(torch.tensor(X))).numpy()

    return X, R, C
    