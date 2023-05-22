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
num_runs = 3
num_epochs = 2
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

# class ReLUBatchNormNet(nn.Module):
#     def __init__(self):
#         super(ReLUBatchNormNet, self).__init__()
#         self.layer1 = nn.Linear(n0, n1)
#         self.layer2 = nn.Linear(n1, n2)
#         self.layer3 = nn.Linear(n2, n3)
#         self.bn1 = nn.BatchNorm1d(n1)
#         self.bn2 = nn.BatchNorm1d(n2)


#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         x = self.layer1(x)
#         x = activation(self.bn1(x))
#         x = self.layer2(x)
#         x = activation(self.bn2(x))
#         x = output_normalisation(self.layer3(x))
#         return x



###
# Running experiments
###
FileLog = Log()
ModelList = []
for run in range(num_runs):
    runlogA = RunLog("mnist",network_size,"relu",False,False,experiment_number=experiment_number)
    runlogB = RunLog("mnist",network_size,"relu",False,True,experiment_number=experiment_number)
    runlogC = RunLog("mnist",network_size,"relu",True,True,experiment_number=experiment_number)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    X, Y = sample_dataset(train_dataset, train_loader, data_sample_size)
    Xtest, Ytest = sample_dataset(test_dataset, test_loader, data_sample_size)

    modelDefault = ReLUNet().to(device) # no reinit + batchnorm
    modelRescale = ReLUNet().to(device) # reinit + batchnorm
    modelReinit = ReLUNet().to(device) # reinit + our rescaling

    c_default = reinitialise_network(modelDefault, X, Y,
                                     return_cost_vector = True,
                                     adjust_regions = False,
                                     adjust_variance = False)
    modelDefault = modelDefault.to(device)
    c_default_test = reinitialise_network(modelDefault, Xtest, Ytest,
                                          return_cost_vector = True,
                                          adjust_regions = False,
                                          adjust_variance = False)
    runlogA.record_cost_vector(0,0,[c_default,c_default_test])

    c_rescale = reinitialise_network(modelRescale, X, Y,
                                     return_cost_vector = True,
                                     adjust_regions = False,
                                     adjust_variance = True)
    modelRescale = modelRescale.to(device)
    c_rescale_test = reinitialise_network(modelRescale, Xtest, Ytest,
                                          return_cost_vector = True,
                                          adjust_regions = False,
                                          adjust_variance = False)
    runlogB.record_cost_vector(0,0,[c_rescale,c_rescale_test])

    print("run ",run+1," of ",num_runs,": reinitialising")
    c_reinit = reinitialise_network(modelReinit, X, Y,
                                    return_cost_vector=True,
                                    adjust_regions = True,
                                    adjust_variance = True)
    modelReinit = modelReinit.to(device)
    c_reinit_test = reinitialise_network(modelReinit, Xtest, Ytest,
                                         return_cost_vector = True,
                                         adjust_regions = False,
                                         adjust_variance = False)
    runlogC.record_cost_vector(0,0,[c_reinit,c_reinit_test])

    criterion = nn.CrossEntropyLoss()
    optimizerDefault = torch.optim.Adam(modelDefault.parameters())
    optimizerRescale = torch.optim.Adam(modelRescale.parameters())
    optimizerReinit = torch.optim.Adam(modelReinit.parameters())

    n_total_steps = len(train_loader)

    imagesTest = torch.tensor([],dtype=torch.long)
    labelsTest = torch.tensor([],dtype=torch.long)
    for images, labels in test_loader:
        imagesTest = torch.cat((imagesTest,images))
        labelsTest = torch.cat((labelsTest,labels))

    imagesTest = imagesTest.to(device)
    labelsTest = labelsTest.to(device)

    modelListInit = [deepcopy_network(modelDefault),
                     deepcopy_network(modelRescale),
                     deepcopy_network(modelReinit)]

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputsDefault = modelDefault(images)
            lossDefault = criterion(outputsDefault, labels)
            outputsRescale = modelRescale(images)
            lossRescale = criterion(outputsRescale, labels)
            outputsReinit = modelReinit(images)
            lossReinit = criterion(outputsReinit, labels)

            outputsDefaultTest = modelDefault(imagesTest)
            lossDefaultTest = criterion(outputsDefaultTest, labelsTest)
            outputsRescaleTest = modelRescale(imagesTest)
            lossRescaleTest = criterion(outputsRescaleTest, labelsTest)
            outputsReinitTest = modelReinit(imagesTest)
            lossReinitTest = criterion(outputsReinitTest, labelsTest)

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

            # in each epoch, the step counter i goes from 0 to 599
            if (i+1) % 60 == 0:
                # print("run ",run+1,"/ ",num_runs,";  epoch ",epoch+1," / ",num_epochs)
                # print("    logging accuracies at step ",i+1)

                ###
                # log accuracies and losses every 60th step, i.e., 10 times per epoch
                ###
                with torch.no_grad():
                    n_correct_default = 0
                    n_correct_rescale = 0
                    n_correct_reinit = 0
                    n_samples = 0
                    n_class_correct_default = [0 for i in range(10)]
                    n_class_correct_rescale = [0 for i in range(10)]
                    n_class_correct_reinit = [0 for i in range(10)]
                    n_class_samples = [0 for i in range(10)]
                    for images, labels in train_loader:
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

                    acc_default = [round(n_correct_default / n_samples,3)]
                    acc_rescale = [round(n_correct_rescale / n_samples,3)]
                    acc_reinit = [round(n_correct_reinit / n_samples,3)]
                    acc_default += [round(n_class_correct_default[j] / n_class_samples[j],3) for j in range(10)]
                    acc_rescale += [round(n_class_correct_rescale[j] / n_class_samples[j],3) for j in range(10)]
                    acc_reinit += [round(n_class_correct_reinit[j] / n_class_samples[j],3) for j in range(10)]

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

                    acc_default_test = [round(n_correct_default / n_samples,3)]
                    acc_rescale_test = [round(n_correct_rescale / n_samples,3)]
                    acc_reinit_test = [round(n_correct_reinit / n_samples,3)]
                    acc_default_test += [round(n_class_correct_default[j] / n_class_samples[j],3) for j in range(10)]
                    acc_rescale_test += [round(n_class_correct_rescale[j] / n_class_samples[j],3) for j in range(10)]
                    acc_reinit_test += [round(n_class_correct_reinit[j] / n_class_samples[j],3) for j in range(10)]

                    runlogA.record_accuracies(epoch,i+1,[acc_default,acc_default_test])
                    runlogB.record_accuracies(epoch,i+1,[acc_rescale,acc_rescale_test])
                    runlogC.record_accuracies(epoch,i+1,[acc_reinit,acc_reinit_test])
                    runlogA.record_losses(epoch,i+1,[round(lossDefault.item(),3),round(lossDefaultTest.item(),3)])
                    runlogB.record_losses(epoch,i+1,[round(lossRescale.item(),3),round(lossRescaleTest.item(),3)])
                    runlogC.record_losses(epoch,i+1,[round(lossReinit.item(),3),round(lossReinitTest.item(),3)])

            ###
            # log costs every 150th step, i.e., 4 times per epoch
            ###
            if (i+1) % 150 == 0:
                print("run ",run+1,"/ ",num_runs,";  epoch ",epoch+1," / ",num_epochs)
                print("    logging linear regions at step ",i+1)

                cost_default = reinitialise_network(modelDefault, X, Y, True, False, False)
                cost_rescale = reinitialise_network(modelRescale, X, Y, True, False, False)
                cost_reinit = reinitialise_network(modelReinit, X, Y, True, False, False)
                cost_default_test = reinitialise_network(modelDefault, Xtest, Ytest, True, False, False)
                cost_rescale_test = reinitialise_network(modelRescale, Xtest, Ytest, True, False, False)
                cost_reinit_test = reinitialise_network(modelReinit, Xtest, Ytest, True, False, False)
                runlogA.record_cost_vector(epoch,i+1,[cost_default,cost_default_test])
                runlogB.record_cost_vector(epoch,i+1,[cost_rescale,cost_rescale_test])
                runlogC.record_cost_vector(epoch,i+1,[cost_reinit,cost_reinit_test])

    modelListTrained = [deepcopy_network(modelDefault),
                        deepcopy_network(modelRescale),
                        deepcopy_network(modelReinit)]

    FileLog.add_runlog(runlogA)
    FileLog.add_runlog(runlogB)
    FileLog.add_runlog(runlogC)

    modelList.append([[modelListInit[0],modelListTrained[0]],
                      [modelListInit[1],modelListTrained[1]],
                      [modelListInit[2],modelListTrained[2]]])

FileLog.save(experiment_number)
torch.save(modelList,str(experiment_number)+'.pt')
print("Experiment number: ", experiment_number)
