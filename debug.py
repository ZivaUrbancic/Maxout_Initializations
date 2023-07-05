###
# Prep work
###
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from log_classes import *
exec(open("initialisation.py").read())
np.set_printoptions(threshold=np.inf)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import warnings
warnings.filterwarnings('error')

###
# Experiment hyperparameters
###
experiment_number = random.randint(0,999999999)
num_runs = 1
num_epochs = 1
batch_size = 100
learning_rate = 0.001
dataset = "MNIST"
network_size = "small" # "small" or "large"


###
# Deepcopy hack
###
def deepcopy_network(model):
    torch.save(model,'hunter2.pt')
    return torch.load('hunter2.pt')



# TODO: make sure the following numbers make sense
# small size = largest network size where default initialisation leads to bad accuracy
# large size = smallest network size where default initialisation leads to good accuracy
if dataset=="MNIST" and network_size=="small":
    n0 = 28*28 # = network_size of input
    n1 = 10
    n2 = 10
    n3 = 10 # = network_size of output
    data_sample_size = 3000
elif dataset=="MNIST" and network_size=="large":
    n0 = 28*28 # = network_size of input
    n1 = 32
    n2 = 16
    n3 = 10 # = network_size of output
    data_sample_size = 3000
elif dataset=="CIFAR10" and network_size=="small":
    n0 = 32*32*3 # = network_size of input
    n1 = 10
    n2 = 10
    n3 = 10 # = network_size of output
    data_sample_size = 3000
elif dataset=="CIFAR10" and network_size=="large":
    n0 = 32*32*3 # = size of input
    n1 = 32
    n2 = 16
    n3 = 10 # = size of output
    data_sample_size = 3000
else:
    raise NameError("unsupported dataset or size")



###
# Loading Data
###
if dataset == "MNIST":
    # copied from ...
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,),(0.3081,))])
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               download=True,
                                               transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              download=True,
                                              transform=transform)
    # moved below to ensure that data batches are different in each run
    # train_loader = torch.utils.data.DataLoader(trainset,
    #                                           batch_size=batch_size,
    #                                           shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

if dataset == "CIFAR10": # todo
    # copied from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                 train=True,
                                                 download=True,
                                                 transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                train=False,
                                                download=True,
                                                transform=transform)
    # moved below to ensure that data batches are different in each run
    # train_loader = torch.utils.data.DataLoader(trainset,
    #                                           batch_size=batch_size,
    #                                           shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



###
# Defining Network
###
activation = nn.ReLU()
output_normalisation = nn.LogSoftmax(dim=-1)
class ReLUNet(nn.Module):
    def __init__(self):
        super(ReLUNet, self).__init__()
        self.layer1 = nn.Linear(n0, n1)
        self.act1 = activation
        self.layer2 = nn.Linear(n1, n2)
        self.act2 = activation
        self.layer3 = nn.Linear(n2, n3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        x = self.layer3(x)
        return output_normalisation(x)


###
# Running experiments
###
for randSeed in range(2**28,2**30):
    print("randSeed: ",randSeed)
    np.random.seed(randSeed)

    FileLog = Log()
    ModelANetworks = []
    ModelCNetworks = []

    runlogA = RunLog("mnist",network_size,"relu",False,False,experiment_number=experiment_number)
    runlogC = RunLog("mnist",network_size,"relu",True,True,experiment_number=experiment_number)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    X, Y = sample_dataset(train_dataset, train_loader, data_sample_size)
    Xtest, Ytest = sample_dataset(test_dataset, test_loader, data_sample_size)

    modelDefault = ReLUNet().to(device) # no reinit + batchnorm
    modelReinit = ReLUNet().to(device) # reinit + our rescaling

    c_default = reinitialise_network(modelDefault, X, Y,
                                     return_cost_vector = True,
                                     adjust_regions = False)
    modelDefault = modelDefault.to(device)
    c_default_test = reinitialise_network(modelDefault, Xtest, Ytest,
                                          return_cost_vector = True,
                                          adjust_regions = False)
    runlogA.record_cost_vector(0,0,[c_default,c_default_test])

    c_reinit = reinitialise_network(modelReinit, X, Y,
                                    return_cost_vector=True,
                                    adjust_regions = True)
    modelReinit = modelReinit.to(device)
    c_reinit_test = reinitialise_network(modelReinit, Xtest, Ytest,
                                         return_cost_vector = True,
                                         adjust_regions = False)
    runlogC.record_cost_vector(0,0,[c_reinit,c_reinit_test])

    criterion = nn.CrossEntropyLoss()
    optimizerDefault = torch.optim.Adam(modelDefault.parameters())
    optimizerReinit = torch.optim.Adam(modelReinit.parameters())

    n_total_steps = len(train_loader)

    imagesTest = torch.tensor([],dtype=torch.long)
    labelsTest = torch.tensor([],dtype=torch.long)
    for images, labels in test_loader:
        imagesTest = torch.cat((imagesTest,images))
        labelsTest = torch.cat((labelsTest,labels))

    imagesTest = imagesTest.to(device)
    labelsTest = labelsTest.to(device)

    ModelARunNetworks = [deepcopy_network(modelDefault)]
    ModelCRunNetworks = [deepcopy_network(modelReinit)]

    epoch = 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputsDefault = modelDefault(images)
        lossDefault = criterion(outputsDefault, labels)
        outputsReinit = modelReinit(images)
        lossReinit = criterion(outputsReinit, labels)

        outputsDefaultTest = modelDefault(imagesTest)
        lossDefaultTest = criterion(outputsDefaultTest, labelsTest)
        outputsReinitTest = modelReinit(imagesTest)
        lossReinitTest = criterion(outputsReinitTest, labelsTest)

        # Backward and optimize
        optimizerDefault.zero_grad()
        lossDefault.backward()
        optimizerDefault.step()
        optimizerReinit.zero_grad()
        lossReinit.backward()
        optimizerReinit.step()

        if lossDefault.item() == np.nan:
            print("lossDefault is nan")
            print("randSeed: ",randSeed)
            assert False
        if lossDefaultTest.item() == np.nan:
            print("lossDefaultTest is nan")
            print("randSeed: ",randSeed)
            assert False
        if lossReinit.item() == np.nan:
            print("lossReinit is nan")
            print("randSeed: ",randSeed)
            assert False
        if lossReinitTest.item() == np.nan:
            print("lossReinitTest is nan")
            print("randSeed: ",randSeed)
            assert False

print("Experiment number: ", experiment_number)
