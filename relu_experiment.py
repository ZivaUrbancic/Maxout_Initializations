import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random

exec(open("mnist.py").read())
exec(open("initialisation.py").read())
np.set_printoptions(threshold=np.inf)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
experiment_number = random.randint(0,999999999)
num_runs = 1
num_epochs = 2
batch_size = 100
learning_rate = 0.001
dataset = "MNIST"

# shamelessly stolen from the web, as is everything else in this file
if dataset == "MNIST":
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
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

if dataset == "???": # todo
    pass

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

activation = nn.ReLU()
output_normalisation = nn.LogSoftmax(dim=-1)

class ReLUNet(nn.Module):
    def __init__(self):
        super(ReLUNet, self).__init__()
        self.layer1 = nn.Linear(28*28, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 10)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = activation(self.layer1(x))
        x = activation(self.layer2(x))
        x = output_normalisation(self.layer3(x))
        return x

modelDefault = ReLUNet().to(device)
print("[relu, ???] (relu or maxout, if maxout what rank, architecture sizes, dataset)",file=open(str(experiment_number)+".log",'+a') # todo

for run in range(num_runs):

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True)
    X, X_labels = sample_MNIST(train_dataset, 3000)
    unit_vecs = np.eye(10)
    R10_labels = np.array([unit_vecs[i] for i in X_labels.int()])

    modelDefault = ReLUNet().to(device)
    modelReinit = ReLUNet().to(device)
    print("run ",run+1," of ",num_runs,": reinitialising")
    c_reinit = reinitialise_ReLU_network(modelReinit, X, R10_labels)
    print([[run],c_reinit],file=open(str(experiment_number)+"_cost_reinit.log",'+a'))

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

                        for j in range(labels.size(0)):
                            label = labels[j]
                            pred_default = predictedDefault[j]
                            if (label == pred_default):
                                n_class_correct_default[label] += 1
                            pred_reinit = predictedReinit[j]
                            if (label == pred_reinit):
                                n_class_correct_reinit[label] += 1
                            n_class_samples[label] += 1

                    acc_default = [100 * n_correct_default / n_samples]
                    acc_reinit = [100 * n_correct_reinit / n_samples]
                    acc_default += [n_class_correct_default[j] / n_class_samples[j] for j in range(10)]
                    acc_reinit += [n_class_correct_reinit[j] / n_class_samples[j] for j in range(10)]

                    print([[run,epoch,i],[lossDefault.item()]],file=open(str(experiment_number)+"_loss_default.log",'+a'))
                    print([[run,epoch,i],[lossReinit.item()]],file=open(str(experiment_number)+"_loss_reinit.log",'+a'))
                    print([[run,epoch,i],acc_default],file=open(str(experiment_number)+"_acc_default.log",'+a'))
                    print([[run,epoch,i],acc_reinit],file=open(str(experiment_number)+"_acc_reinit.log",'+a'))
