###
# Old code that should not be used anymore
###

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
