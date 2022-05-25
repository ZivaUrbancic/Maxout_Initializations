###
# Prep work
###
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
exec(open("mnist.py").read())
exec(open("initialisation.py").read())
np.set_printoptions(threshold=np.inf)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


###
# Experiment hyperparameters
###
experiment_number = random.randint(0,999999999)
num_runs = 2
num_epochs = 6
batch_size = 100
learning_rate = 0.001
dataset = "MNIST"
network_size = "large" # "small" or "large"



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
        self.layer2 = nn.Linear(n1, n2)
        self.layer3 = nn.Linear(n2, n3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = activation(self.layer1(x))
        x = activation(self.layer2(x))
        x = output_normalisation(self.layer3(x))
        return x

class ReLUBatchNormNet(nn.Module):
    def __init__(self):
        super(ReLUBatchNormNet, self).__init__()
        self.layer1 = nn.Linear(n0, n1)
        self.layer2 = nn.Linear(n1, n2)
        self.layer3 = nn.Linear(n2, n3)
        self.bn1 = nn.BatchNorm1d(n1)
        self.bn2 = nn.BatchNorm1d(n2)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = activation(self.bn1(x))
        x = self.layer2(x)
        x = activation(self.bn2(x))
        x = output_normalisation(self.layer3(x))
        return x



###
# Running experiments
###
modelDefault = ReLUNet().to(device)
print("activation: relu\n",
      "rank: N/A\n",
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

    modelDefault = ReLUBatchNormNet().to(device) # no reinit + batchnorm
    modelRescale = ReLUBatchNormNet().to(device) # reinit + batchnorm
    modelReinit = ReLUNet().to(device) # reinit + our rescaling

    reinitialise_network(modelRescale, X, Y, adjust_regions = True, adjust_variance = False)
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
