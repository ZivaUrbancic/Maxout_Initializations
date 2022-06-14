###
# Old code that should not be used anymore
###


def fix_variance(X, weights, biases):
    Xvar = np.var(X, axis=0)
    scale_factor = np.reciprocal(np.sqrt(Xvar))
    weights = weights * scale_factor.reshape(len(scale_factor), 1) #np.matmul(np.diag(scale_factor), weights)
    biases = np.multiply(scale_factor, biases)
    return weights, biases


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


def reinitialise_conv2d_layer(child, X, Y, R = False, C = False,
                              adjust_regions = True,
                              adjust_variance = True):

    if type(R) == bool or type(C) == bool:
        N = X.shape[0] # number of data points
        assert R == False and C == False
        R = initialise_region_matrix(N)
        C = initialise_costs_vector(N)

    if adjust_regions == False and adjust_variance == False:
        return X, R, C

    child1 = nn.Conv2d(child.in_channels,
                       child.out_channels,
                       child.kernel_size,
                       child.stride,
                       child.padding,
                       child.dilation,
                       child.groups,
                       False,
                       child.padding_mode)
    if type(X) == np.ndarray:
        X = torch.from_numpy(X)

    X1 = child1(X).detach().numpy()
    X2 = X1.mean(axis = (-2,-1))

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
            print("reinitialising channel",k)
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

    # # Crop the image to a space of dimension c0 = width * height of kernel
    # h, w = child.kernel_size
    # c0 = w*h

    # # Use a new convolutional layer which copies the hyperparameters of
    # # child but with weights that project the image onto the i,j th component
    # crop_conv = nn.Conv2d(child.in_channels,
    #                       c0 * child.in_channels,
    #                       child.kernel_size,
    #                       stride = child.stride,
    #                       padding = child.padding,
    #                       dilation = child.dilation,
    #                       groups = child.groups,
    #                       bias = False,
    #                       padding_mode = child.padding_mode)
    # W = torch.zeros(crop_conv.weight.shape)
    # for channel in range(child.in_channels):
    #     for y in range(h):
    #         for x in range(w):
    #             W[channel*c0 + y*w + x, channel, y, x] = 1.

    # with torch.no_grad():
    #     X_cropped = crop_conv(torch.tensor(X)).numpy()

    # # Find width and height of the image
    # Width, Height = X.shape[-2:]

    # X_ = []

    # # Crop the images to the shape of the kernel (for each out channel)
    # for k in range(child.in_channels):
    #     # Crop each image for each kernel translation, and put them all in
    #     # one big list of length |X| * (W - w + 1) * (H - h + 1)
    #     crops = []
    #     for i in range(Height - h + 1):
    #         for j in range(Width - w + 1):
    #             for point in range(X.shape[0]):
    #                 crops.append([X_cropped[point,k*c0 : (k+1)*c0][p,i,j]
    #                        for p in range(c0)])

    #     X_.append(crops)

    # # Create reshaped X and Y data for the new cropped images
    # X_ = np.concatenate(X_, axis = 1)
    # Y_ = np.tile(Y, (Height - h + 1)*(Width - w + 1))

    # # Create a linear ReLU layer and reinitialise with the cropped data
    # c0_child = nn.Linear(c0*child.in_channels, child.out_channels)
    # c0_child.weight = nn.Parameter(child.weight.reshape(child.out_channels,
    #                                                     c0*child.in_channels))
    # c0_child.bias = child.bias
    # reinitialise_relu_layer(c0_child, X_, Y_, adjust_regions = adjust_regions, adjust_variance = adjust_variance)

    # # Reshape the reinitialised weights as a convolutional weight tensor
    # # and use as weights for the original child
    # reshaped_weights = c0_child.weight.reshape(child.weight.shape)
    # child.weight = nn.Parameter(reshaped_weights)
    # child.bias = c0_child.bias



    # compute image of the dataset under the adjusted parameters
    with torch.no_grad():
        X = nn.ReLU()(child(X)).numpy()

    return X, R, C



# loads N points from each class from a dataset of class 'torchvision.datasets'
# DO NOT USE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def sample_dataset_flatten_and_floatify(dataset,N,random=False):
    indices = np.arange(len(dataset.targets))
    if random:
        np.random.shuffle(indices)
    targets = [target for target in dataset.class_to_idx.values()]
    targets_counter = np.zeros(len(targets))
    X = []
    Xtargets = []

    for i in indices:
        for j in targets:
            if targets[j] == dataset.targets[i] and targets_counter[j]<N:
                x = dataset.data[i]
                # make x into a numpy.ndarray, if it is not
                if type(x)==type(torch.Tensor(1)):
                    x = x.numpy()
                X += [x.flatten()]
                Xtargets += [dataset.targets[i]]
                targets_counter[j] += 1
        if len(Xtargets) == N*len(targets):
            break

    return np.array(X).astype('float32'), Xtargets

def maxout_activation(weights_and_biases):
    functions = []
    for i in range(weights_and_biases.shape[0]):
        functions += [linear(weights_and_biases[i])]
    return functions

def maxout_activation(weights_and_biases):
    functions = []
    for i in range(weights_and_biases.shape[0]):
        functions += [linear(weights_and_biases[i])]
    return functions

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

def largest_regions_have_positive_cost(C, k):
    return k_th_largest_region_cost(C, k) > 0

# model: Maxout network as in maxout.py
# X: training data
# Y: training targets
def reinitialise_Maxout_network(model, X, Y):
    # flatten each datapoint in case input is multidimational
    # e.g. 2D images in MNIST and CIFAR10
    if len(X.shape)>2:
        X = np.array([x.flatten() for x in X])
    # list of sublayers, layer i = sublayers i, i+1, ..., i+maxout_rank-1
    Sublayers = [sublayer for sublayer in model.children()]
    maxout_rank = model.maxout_rank
    N = X.shape[0] # number of data points
    R = initialise_region_matrix(N)
    C = initialise_costs_vector(N)
    number_of_classes=max(Y)+1

    # If using L2 cost:
    #Y = class_to_unit_vector(Y)

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
                                                X, Y, CE_region_cost,
                                                number_of_classes)
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
                                                X, Y, CE_region_cost,
                                                number_of_classes)
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
    number_of_classes=max(Y)+1

    # If using L2 cost:
    Y = class_to_unit_vector(Y)


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
                                                X, Y, CE_region_cost,
                                                number_of_classes)
                WB += [wb[1]] # wb[0] contains only zeroes
                if not largest_regions_have_positive_cost(C,0):
                    stage = 2

            # stage 2:
            # keep existing parameters until the end of the reinitialisation
            else:
                print("keeping layer ", l, " unit ", k)
                w = layer.weight[k,:].detach().numpy()
                b = layer.bias[k].detach().numpy()
                WB += [np.append([w], [b])]

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
