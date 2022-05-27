###
# Prep work
###
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
exec(open("../initialisation.py").read())
np.set_printoptions(threshold=np.inf)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



###
# Experiment hyperparameters
###
experiment_number = random.randint(0,999999999)
num_runs = 1
num_epochs = 2
batch_size = 100
learning_rate = 0.001
dataset = "MNIST"
network_size = "small" # "small" or "large"
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
    train_dataset = torchvision.datasets.CIFAR10(root='../data',
                                                 train=True,
                                                 download=True,
                                                 transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='../data',
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
        x = x.view(x.size(0), -1) # size: batch_size * n0
        X = torch.stack( [self.lay1lin1(x),
                          self.lay1lin2(x),
                          self.lay1lin3(x),
                          self.lay1lin4(x),
                          self.lay1lin5(x),
                          self.lay1lin6(x),
                          self.lay1lin7(x)])
                                  # size: maxout_rank * batch_size * n1
        x,_ = torch.max(X,0)      # size: batch_size * n1
        X = torch.stack( [self.lay2lin1(x),
                          self.lay2lin2(x),
                          self.lay2lin3(x),
                          self.lay2lin4(x),
                          self.lay2lin5(x),
                          self.lay2lin6(x),
                          self.lay2lin7(x)])
                                  # size: maxout_rank * batch_size * n2
        x,_ = torch.max(X,0)      # size: batch_size * n2
        X = torch.stack( [self.lay3lin1(x),
                          self.lay3lin2(x),
                          self.lay3lin3(x),
                          self.lay3lin4(x),
                          self.lay3lin5(x),
                          self.lay3lin6(x),
                          self.lay3lin7(x)])
                                  # size: maxout_rank * batch_size * n3
        x,_ = torch.max(X,0)      # size: batch_size * n3
        # x = mySoftmax(x) # wth does this make loss worse?
        return x

class MaxoutFlexNet(nn.Module):
    def __init__(self):
        super(MaxoutFlexNet, self).__init__()
        self.lay1 = nn.ModuleList([nn.Linear(n0, n1) for i in range(network_rank)])
        self.lay2 = nn.ModuleList([nn.Linear(n1, n2) for i in range(network_rank)])
        self.lay3 = nn.ModuleList([nn.Linear(n2, n3) for i in range(network_rank)])
        self.maxout_rank = network_rank


    def forward(self, x):
        x = x.view(x.size(0), -1) # size: batch_size * n0
        X = torch.stack( [lin(x) for lin in self.lay1] )
                                  # size: network_rank * batch_size * n1
        x,_ = torch.max(X,0)      # size: batch_size * n1
        X = torch.stack( [lin(x) for lin in self.lay2] )
                                  # size: network_rank * batch_size * n2
        x,_ = torch.max(X,0)      # size: batch_size * n2
        X = torch.stack( [lin(x) for lin in self.lay3] )
                                  # size: network_rank * batch_size * n3
        x,_ = torch.max(X,0)      # size: batch_size * n3
        # x = mySoftmax(x) # wth does this make loss worse?
        return x

###
# Running experiments
###
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

    ###
    # Definition + Reinitialisation
    # (set models to None if not needed, e.g. 'modelC = None'
    ###
    print("run ",run+1," of ",num_runs,": reinitialising")
    modelA = MaxoutNet().to(device)     # maxout_rank hardcoded
    # reinitialise_network(modelA, X, Y, adjust_regions = True, adjust_variance = False)
    # modelA = modelA.to(device)

    modelB = MaxoutFlexNet().to(device) # maxout_rank flexible
    # reinitialise_network(modelB, X, Y, adjust_regions = True, adjust_variance = False)
    # modelB = modelB.to(device)

    modelC = None                       #
    # c_reinit = reinitialise_network(modelC, X, Y, adjust_regions = True, adjust_variance = False)
    # modelC = modelC.to(device)
    # print([run,c_reinit],file=open(str(experiment_number)+"_cost_reinit.log",'+a'))


    ###
    # Training + Logging
    ###
    criterion = nn.CrossEntropyLoss()
    if modelA != None:
        optimizerA = torch.optim.SGD(modelA.parameters(), lr=learning_rate)
    if modelB != None:
        optimizerB = torch.optim.SGD(modelB.parameters(), lr=learning_rate)
    if modelC != None:
        optimizerC = torch.optim.SGD(modelC.parameters(), lr=learning_rate)

    log_loss_A = [] # for logging purposes
    log_loss_B = []
    log_loss_C = []

    print("run ",run+1," of ",num_runs,": training")
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        print("  epoch ",epoch+1," of ",num_epochs)
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            if modelA != None:
                # Forward pass
                outputsA = modelA(images)
                lossA = criterion(outputsA, labels)
                # Backward and optimize
                optimizerA.zero_grad()
                lossA.backward()
                optimizerA.step()
            if modelB != None:
                # Forward pass
                outputsB = modelB(images)
                lossB = criterion(outputsB, labels)
                # Backward and optimize
                optimizerB.zero_grad()
                lossB.backward()
                optimizerB.step()
            if modelC != None:
                # Forward pass
                outputsC = modelC(images)
                lossC = criterion(outputsC, labels)
                # Backward and optimize
                optimizerC.zero_grad()
                lossC.backward()
                optimizerC.step()

            ###
            # Bookkeeping (only every 10 steps)
            ###
            if (i+1) % 10 > 0:
                continue

            with torch.no_grad():
                n_samples = 0
                n_class_samples = [0 for i in range(10)]
                n_correct_A = 0
                n_class_correct_A = [0 for i in range(10)]
                n_correct_B = 0
                n_class_correct_B = [0 for i in range(10)]
                n_correct_C = 0
                n_class_correct_C = [0 for i in range(10)]
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    if modelA != None:
                        outputsA = modelA(images)
                        _, predictedA = torch.max(outputsA, 1)
                        n_correct_A += (predictedA == labels).sum().item()
                    if modelB != None:
                        outputsB = modelB(images)
                        _, predictedB = torch.max(outputsB, 1)
                        n_correct_B += (predictedB == labels).sum().item()
                    if modelC != None:
                        outputsC = modelC(images)
                        _, predictedC = torch.max(outputsC, 1)
                        n_correct_C += (predictedC == labels).sum().item()

                    n_samples += labels.size(0)
                    for j,label in enumerate(labels):
                        if modelA != None and label == predictedA[j]:
                            n_class_correct_A[label] += 1
                        if modelB != None and label == predictedB[j]:
                            n_class_correct_B[label] += 1
                        if modelC != None and label == predictedC[j]:
                            n_class_correct_C[label] += 1
                        n_class_samples[label] += 1

                if modelA != None:
                    acc_A = [100 * n_correct_A / n_samples]
                    acc_A += [n_class_correct_A[j] / n_class_samples[j] for j in range(10)]
                    print([[run,epoch,i],[lossA.item()]],file=open(str(experiment_number)+"_loss_A.log",'+a'))
                    print([[run,epoch,i],acc_A],file=open(str(experiment_number)+"_acc_A.log",'+a'))

                if modelB != None:
                    acc_B = [100 * n_correct_B / n_samples]
                    acc_B += [n_class_correct_B[j] / n_class_samples[j] for j in range(10)]
                    print([[run,epoch,i],[lossB.item()]],file=open(str(experiment_number)+"_loss_B.log",'+a'))
                    print([[run,epoch,i],acc_B],file=open(str(experiment_number)+"_acc_B.log",'+a'))

                if modelC != None:
                    acc_C = [100 * n_correct_C / n_samples]
                    acc_C += [n_class_correct_C[j] / n_class_samples[j] for j in range(10)]
                    print([[run,epoch,i],[lossC.item()]],file=open(str(experiment_number)+"_loss_C.log",'+a'))
                    print([[run,epoch,i],acc_C],file=open(str(experiment_number)+"_acc_C.log",'+a'))
