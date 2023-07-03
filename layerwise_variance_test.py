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


###
# Experiment hyperparameters
###
experiment_number = random.randint(0,999999999)
num_runs = 24
num_epochs = 24
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

train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

model = ReLUNet()
X, Y = sample_dataset(train_dataset, train_loader, data_sample_size)
if len(X.shape)>2:
    X = np.array([x.flatten() for x in X])

v = layerwise_deviation(model, X)

children = list(model.children())

child = children[0]
print(child.weight)
print(child.bias)

with torch.no_grad():
    Xtemp = nn.ReLU()(child(torch.tensor(X))).numpy()
fix_child_deviation(child, Xtemp, v[0])

print(child.weight)
print(child.bias)