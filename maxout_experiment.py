import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import json

exec(open("mnist.py").read())
exec(open("initialisation.py").read())

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# shamelessly stolen from the web, as is everything else in this file
transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))
                                ])


# MNIST: 60000 28x28 color images in 10 classes
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           download=True,
                                           transform=transform)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          download=True,
                                          transform=transform)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
# dataiter = iter(train_loader)
# images, labels = dataiter.next()
# show images
# imshow(torchvision.utils.make_grid(images))

mySoftmax = torch.nn.Softmax(dim=0)

class MaxoutNet(nn.Module):
    def __init__(self):
        super(MaxoutNet, self).__init__()
        self.lay1lin1 = nn.Linear(28*28, 11)
        self.lay1lin2 = nn.Linear(28*28, 11)
        self.lay1lin3 = nn.Linear(28*28, 11)
        self.lay2lin1 = nn.Linear(11, 10)
        self.lay2lin2 = nn.Linear(11, 10)
        self.lay2lin3 = nn.Linear(11, 10)
        self.lay3lin1 = nn.Linear(10, 10)
        self.lay3lin2 = nn.Linear(10, 10)
        self.lay3lin3 = nn.Linear(10, 10)
        self.maxout_rank = 3


    def forward(self, x):
        x = x.view(x.size(0), -1) # I do not understand this black magic, x is not a single image, but a tensor of images. How does this code work?
        x = x.unsqueeze(0) # make vector of length 784 into 1*784 matrix
        X = torch.cat( (self.lay1lin1(x),self.lay1lin2(x),self.lay1lin3(x)), 0)
              # concatenate output vectors into matrix (row-wise by default)
              # size: rank * width layer 1
        x,dummy = torch.max(X,0)
              # go through each column and compute max
              # size: 1 * width layer 1
        x = x.unsqueeze(0)
        X = torch.cat( (self.lay2lin1(x),self.lay2lin2(x),self.lay2lin3(x)), 0)
              # concatenate output vectors into matrix (row-wise by default)
              # size: rank * width layer 2
        x,dummy = torch.max(X,0)
              # go through each column and compute max
              # size: 1 * width layer 2
        x = x.unsqueeze(0)
        X = torch.cat( (self.lay3lin1(x),self.lay3lin2(x),self.lay3lin3(x)), 0)
              # concatenate output vectors into matrix (row-wise by default)
              # size: rank * width layer 2
        x,dummy = torch.max(X,0)
              # go through each column and compute max
              # size: 1 * width layer 2
        # x = mySoftmax(x) # wth does this make loss worse?
        return x

experiment_number = random.randint(0,999999999)

num_runs = 10

for run in range(num_runs):

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True)
    X, X_labels = sample_MNIST(train_dataset, 3000)
    unit_vecs = np.eye(10)
    R10_labels = np.array([unit_vecs[i] for i in X_labels.int()])

    modelDefault = MaxoutNet().to(device)
    modelReinit = MaxoutNet().to(device)
    print("run ",run+1," of ",num_runs,": reinitialising")
    reinitialise_Maxout_network(modelReinit, X, R10_labels)

    criterion = nn.CrossEntropyLoss()
    optimizerDefault = torch.optim.SGD(modelDefault.parameters(), lr=learning_rate)
    optimizerReinit = torch.optim.SGD(modelReinit.parameters(), lr=learning_rate)

    log_loss_default = []
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
            outputsReinit = modelReinit(images)
            lossReinit = criterion(outputsReinit, labels)

            # Backward and optimize
            optimizerDefault.zero_grad()
            lossDefault.backward()
            optimizerDefault.step()
            optimizerReinit.zero_grad()
            lossReinit.backward()
            optimizerReinit.step()

            if (i+1) % 10 == 0:
                log_loss_default += [lossDefault.item()]
                log_loss_reinit += [lossReinit.item()]

    print("run ",run+1," of ",num_runs,": writing")
    with torch.no_grad():
        n_correct_default = 0
        n_correct_reinit = 0
        n_samples = 0
        n_class_correct_default = [0 for i in range(10)]
        n_class_correct_reinit = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputsDefault = modelDefault(images)
            outputsReinit = modelReinit(images)
            _, predictedDefault = torch.max(outputsDefault, 1)
            _, predictedReinit = torch.max(outputsReinit, 1)
            n_samples += labels.size(0)
            n_correct_default += (predictedDefault == labels).sum().item()
            n_correct_reinit += (predictedReinit == labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i]
                pred_default = predictedDefault[i]
                if (label == pred_default):
                    n_class_correct_default[label] += 1
                pred_reinit = predictedReinit[i]
                if (label == pred_reinit):
                    n_class_correct_reinit[label] += 1
                n_class_samples[label] += 1

        acc_default = [n_correct_default / n_samples]
        acc_reinit = [100.0 * n_correct_reinit / n_samples]

        for i in range(10):
            acc_default += [n_class_correct_default[i] / n_class_samples[i]]
            acc_reinit += [n_class_correct_reinit[i] / n_class_samples[i]]

        print(log_loss_default,file=open(str(experiment_number)+"_loss_default.log",'+a'))
        print(log_loss_reinit,file=open(str(experiment_number)+"_loss_reinit.log",'+a'))
        print(acc_default,file=open(str(experiment_number)+"_acc_default.log",'+a'))
        print(acc_reinit,file=open(str(experiment_number)+"_acc_reinit.log",'+a'))
