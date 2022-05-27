###
# Prep work
###
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
exec(open("initialisation.py").read())
np.set_printoptions(threshold=np.inf)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



###
# Experiment hyperparameters
###
experiment_number = random.randint(0,999999999)
num_runs = 12
num_epochs = 12
batch_size = 100
learning_rate = 0.001
dataset = "MNIST"
network_size = "large" # "small" or "large"
network_rank = 7 # WARNING: does not change network below, adjust by hand



# TODO: make sure the following numbers make sense
# small size = largest network size where default initialisation leads to bad accuracy
# large size = smallest network size where default initialisation leads to good accuracy
if dataset=="MNIST" and network_size=="small":
    n0 = 28*28 # = network_size of input
    n1 = 10
    n2 = 10
    n3 = 10 # = network_size of output
    data_sample_size = 10000
elif dataset=="MNIST" and network_size=="large":
    n0 = 28*28 # = network_size of input
    n1 = 32
    n2 = 16
    n3 = 10 # = network_size of output
    data_sample_size = 10000
elif dataset=="CIFAR10" and network_size=="small":
    n0 = 32*32*3 # = network_size of input
    n1 = 32*32*3
    n2 = 1024
    n3 = 256
    n4 = 64
    n5 = 10 # = network_size of output
    data_sample_size = 10000
elif dataset=="CIFAR10" and network_size=="large":
    n0 = 32*32*3 # = size of input
    n1 = 32
    n2 = 16
    n3 = 10 # = size of output
    data_sample_size = 10000
else:
    raise NameError("unsupported dataset or size")



###
# Loading Data
###
if dataset == "MNIST":
    # copied from ...
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,),(0.3081,))])
    train_dataset = torchvision.datasets.MNIST(root='../data',
                                               train=True,
                                               download=True,
                                               transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='../data',
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
mySoftmax = torch.nn.Softmax(dim=0)
class MaxoutNet(nn.Module):
    def __init__(self):
        super(MaxoutNet, self).__init__()
        self.lay1lin1 = nn.Linear(n0, n1)
        self.lay1lin2 = nn.Linear(n0, n1)
        self.lay1lin3 = nn.Linear(n0, n1)
        self.lay1lin4 = nn.Linear(n0, n1)
        self.lay1lin5 = nn.Linear(n0, n1)
        self.lay1lin6 = nn.Linear(n0, n1)
        self.lay1lin7 = nn.Linear(n0, n1)
        self.lay2lin1 = nn.Linear(n1, n2)
        self.lay2lin2 = nn.Linear(n1, n2)
        self.lay2lin3 = nn.Linear(n1, n2)
        self.lay2lin4 = nn.Linear(n1, n2)
        self.lay2lin5 = nn.Linear(n1, n2)
        self.lay2lin6 = nn.Linear(n1, n2)
        self.lay2lin7 = nn.Linear(n1, n2)
        self.lay3lin1 = nn.Linear(n2, n3)
        self.lay3lin2 = nn.Linear(n2, n3)
        self.lay3lin3 = nn.Linear(n2, n3)
        self.lay3lin4 = nn.Linear(n2, n3)
        self.lay3lin5 = nn.Linear(n2, n3)
        self.lay3lin6 = nn.Linear(n2, n3)
        self.lay3lin7 = nn.Linear(n2, n3)
        self.maxout_rank = 7


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = x.unsqueeze(0) # make vector of length n0 into 1*n0 matrix
        X = torch.cat( (self.lay1lin1(x),
                        self.lay1lin2(x),
                        self.lay1lin3(x),
                        self.lay1lin4(x),
                        self.lay1lin5(x),
                        self.lay1lin6(x),
                        self.lay1lin7(x)), 0)
              # concatenate output vectors into matrix (row-wise by default)
              # size: rank * width layer 1
        x,dummy = torch.max(X,0)
              # go through each column and compute max
              # size: 1 * width layer 1
        x = x.unsqueeze(0)
        X = torch.cat( (self.lay2lin1(x),
                        self.lay2lin2(x),
                        self.lay2lin3(x),
                        self.lay2lin4(x),
                        self.lay2lin5(x),
                        self.lay2lin6(x),
                        self.lay2lin7(x)), 0)
              # concatenate output vectors into matrix (row-wise by default)
              # size: rank * width layer 2
        x,dummy = torch.max(X,0)
              # go through each column and compute max
              # size: 1 * width layer 2
        x = x.unsqueeze(0)
        X = torch.cat( (self.lay3lin1(x),
                        self.lay3lin2(x),
                        self.lay3lin3(x),
                        self.lay3lin4(x),
                        self.lay3lin5(x),
                        self.lay3lin6(x),
                        self.lay3lin7(x)), 0)
              # concatenate output vectors into matrix (row-wise by default)
              # size: rank * width layer 2
        x,dummy = torch.max(X,0)
              # go through each column and compute max
              # size: 1 * width layer 2
        # x = mySoftmax(x) # wth does this make loss worse?
        return x

class MaxoutBatchnormNet(nn.Module):
    def __init__(self):
        super(MaxoutBatchnormNet, self).__init__()
        self.lay1lin1 = nn.Linear(n0, n1)
        self.lay1lin2 = nn.Linear(n0, n1)
        self.lay1lin3 = nn.Linear(n0, n1)
        self.lay1lin4 = nn.Linear(n0, n1)
        self.lay1lin5 = nn.Linear(n0, n1)
        self.lay1lin6 = nn.Linear(n0, n1)
        self.lay1lin7 = nn.Linear(n0, n1)
        self.lay2lin1 = nn.Linear(n1, n2)
        self.lay2lin2 = nn.Linear(n1, n2)
        self.lay2lin3 = nn.Linear(n1, n2)
        self.lay2lin4 = nn.Linear(n1, n2)
        self.lay2lin5 = nn.Linear(n1, n2)
        self.lay2lin6 = nn.Linear(n1, n2)
        self.lay2lin7 = nn.Linear(n1, n2)
        self.lay3lin1 = nn.Linear(n2, n3)
        self.lay3lin2 = nn.Linear(n2, n3)
        self.lay3lin3 = nn.Linear(n2, n3)
        self.lay3lin4 = nn.Linear(n2, n3)
        self.lay3lin5 = nn.Linear(n2, n3)
        self.lay3lin6 = nn.Linear(n2, n3)
        self.lay3lin7 = nn.Linear(n2, n3)
        self.bn1 = nn.BatchNorm1d(n1)
        self.bn2 = nn.BatchNorm1d(n2)
        self.bn3 = nn.BatchNorm1d(n3)
        self.maxout_rank = 7


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = x.unsqueeze(0) # make vector of length n0 into 1*n0 matrix
        X = torch.cat( (self.bn1(self.lay1lin1(x)),
                        self.bn1(self.lay1lin2(x)),
                        self.bn1(self.lay1lin3(x)),
                        self.bn1(self.lay1lin4(x)),
                        self.bn1(self.lay1lin5(x)),
                        self.bn1(self.lay1lin6(x)),
                        self.bn1(self.lay1lin7(x))), 0)
              # concatenate output vectors into matrix (row-wise by default)
              # size: rank * width layer 1
        x,dummy = torch.max(X,0)
              # go through each column and compute max
              # size: 1 * width layer 1
        x = x.unsqueeze(0)
        X = torch.cat( (self.bn2(self.lay2lin1(x)),
                        self.bn2(self.lay2lin2(x)),
                        self.bn2(self.lay2lin3(x)),
                        self.bn2(self.lay2lin4(x)),
                        self.bn2(self.lay2lin5(x)),
                        self.bn2(self.lay2lin6(x)),
                        self.bn2(self.lay2lin7(x))), 0)
              # concatenate output vectors into matrix (row-wise by default)
              # size: rank * width layer 2
        x,dummy = torch.max(X,0)
              # go through each column and compute max
              # size: 1 * width layer 2
        x = x.unsqueeze(0)
        X = torch.cat( (self.bn3(self.lay3lin1(x)),
                        self.bn3(self.lay3lin2(x)),
                        self.bn3(self.lay3lin3(x)),
                        self.bn3(self.lay3lin4(x)),
                        self.bn3(self.lay3lin5(x)),
                        self.bn3(self.lay3lin6(x)),
                        self.bn3(self.lay3lin7(x))), 0)
              # concatenate output vectors into matrix (row-wise by default)
              # size: rank * width layer 2
        x,dummy = torch.max(X,0)
              # go through each column and compute max
              # size: 1 * width layer 2
        # x = mySoftmax(x) # wth does this make loss worse?
        return x

###
# Running experiments
###
modelDefault = MaxoutNet().to(device)
print("activation: maxout\n",
      "rank:",network_rank,"\n",
      "hidden layers: 2\n",
      "widths:",n0,n1,n2,n3,"\n",
      "dataset:",dataset,"\n",
      "data_sample_size:",data_sample_size,"\n",
      "num_runs:",num_runs,"\n",
      "num_epochs",num_epochs,"\n",
      file=open(str(experiment_number)+".log",'+a'))

for run in range(num_runs):
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    X, Y = sample_dataset(train_dataset, train_loader, data_sample_size)

    modelDefault = MaxoutBatchnormNet().to(device) # batchnorm only
    modelRescale = MaxoutNet().to(device)          # rescale only
    modelReinit = MaxoutBatchnormNet().to(device)  # rescale + batchnorm

    reinitialise_network(modelDefault, X, Y, adjust_regions = True, adjust_variance = False)
    modelDefault = modelDefault.to(device)

    reinitialise_network(modelRescale, X, Y, adjust_regions = True, adjust_variance = True)
    modelRescale = modelRescale.to(device)

    print("run ",run+1," of ",num_runs,": reinitialising")
    c_reinit = reinitialise_network(modelReinit, X, Y, adjust_regions = True, adjust_variance = True)
    modelReinit = modelReinit.to(device)
    print([run,c_reinit],file=open(str(experiment_number)+"_cost_reinit.log",'+a'))

    criterion = nn.CrossEntropyLoss()
    optimizerDefault = torch.optim.SGD(modelDefault.parameters(), lr=learning_rate)
    optimizerRescale = torch.optim.SGD(modelRescale.parameters(), lr=learning_rate)
    optimizerReinit = torch.optim.SGD(modelReinit.parameters(), lr=learning_rate)

    log_loss_default = []
    log_loss_rescale = []
    log_loss_reinit = []

    print("run ",run+1," of ",num_runs,": training")
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        print("  epoch ",epoch+1," of ",num_epochs)
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputsDefault = modelDefault(images)
            lossDefault = criterion(outputsDefault, labels)
            outputsRescale = modelRescale(images)
            lossRescale = criterion(outputsRescale, labels)
            outputsReinit = modelReinit(images)
            lossReinit = criterion(outputsReinit, labels)

            # Backward and optimize
            optimizerDefault.zero_grad()
            lossDefault.backward()
            optimizerDefault.step()
            optimizerRescale.zero_grad()
            lossRescale.backward()
            optimizerRescale.step()
            optimizerReinit.zero_grad()
            lossReinit.backward()
            optimizerReinit.step()

            if (i+1) % 10 == 0:

                with torch.no_grad():
                    n_correct_default = 0
                    n_correct_rescale = 0
                    n_correct_reinit = 0
                    n_samples = 0
                    n_class_correct_default = [0 for i in range(10)]
                    n_class_correct_rescale = [0 for i in range(10)]
                    n_class_correct_reinit = [0 for i in range(10)]
                    n_class_samples = [0 for i in range(10)]
                    for images, labels in test_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputsDefault = modelDefault(images)
                        outputsRescale = modelRescale(images)
                        outputsReinit = modelReinit(images)
                        _, predictedDefault = torch.max(outputsDefault, 1)
                        _, predictedRescale = torch.max(outputsRescale, 1)
                        _, predictedReinit = torch.max(outputsReinit, 1)
                        n_samples += labels.size(0)
                        n_correct_default += (predictedDefault == labels).sum().item()
                        n_correct_rescale += (predictedRescale == labels).sum().item()
                        n_correct_reinit += (predictedReinit == labels).sum().item()

                        for j in range(labels.size(0)):
                            label = labels[j]
                            pred_default = predictedDefault[j]
                            if (label == pred_default):
                                n_class_correct_default[label] += 1
                            pred_rescale = predictedRescale[j]
                            if (label == pred_rescale):
                                n_class_correct_rescale[label] += 1
                            pred_reinit = predictedReinit[j]
                            if (label == pred_reinit):
                                n_class_correct_reinit[label] += 1
                            n_class_samples[label] += 1

                    acc_default = [100 * n_correct_default / n_samples]
                    acc_rescale = [100 * n_correct_rescale / n_samples]
                    acc_reinit = [100 * n_correct_reinit / n_samples]
                    acc_default += [n_class_correct_default[j] / n_class_samples[j] for j in range(10)]
                    acc_rescale += [n_class_correct_rescale[j] / n_class_samples[j] for j in range(10)]
                    acc_reinit += [n_class_correct_reinit[j] / n_class_samples[j] for j in range(10)]

                    print([[run,epoch,i],[lossDefault.item()]],file=open(str(experiment_number)+"_loss_default.log",'+a'))
                    print([[run,epoch,i],[lossRescale.item()]],file=open(str(experiment_number)+"_loss_rescale.log",'+a'))
                    print([[run,epoch,i],[lossReinit.item()]],file=open(str(experiment_number)+"_loss_reinit.log",'+a'))
                    print([[run,epoch,i],acc_default],file=open(str(experiment_number)+"_acc_default.log",'+a'))
                    print([[run,epoch,i],acc_rescale],file=open(str(experiment_number)+"_acc_rescale.log",'+a'))
                    print([[run,epoch,i],acc_reinit],file=open(str(experiment_number)+"_acc_reinit.log",'+a'))
