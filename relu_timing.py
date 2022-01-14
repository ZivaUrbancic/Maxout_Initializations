import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import cProfile


exec(open("mnist.py").read())
exec(open("initialisation.py").read())

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 3
batch_size = 64
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

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

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
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
# imshow(torchvision.utils.make_grid(images))

activation = nn.ReLU()
output_normalisation = nn.LogSoftmax(dim=-1)

class ReLUNet(nn.Module):
    def __init__(self):
        super(ReLUNet, self).__init__()
        self.layer1 = nn.Linear(28*28, 128)
        #self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(32, 10)


    def forward(self, x):
        x = x.view(x.size(0), -1) # I do not understand this black magic, x is not a single image, but a tensor of images. How does this code work?
        x = activation(self.layer1(x))
        x# = activation(self.layer2(x))
        x = output_normalisation(self.layer3(x))
        return x

X, X_labels = sample_MNIST(train_dataset, 6000)

unit_vecs = np.eye(10)
R10_labels = np.array([unit_vecs[i] for i in X_labels.int()])

model = ReLUNet().to(device)

# # For experiments:
# start_time = time.time()
# reinitialise_ReLU_network(model, X, R10_labels)
# print("--- %s seconds ---" % (time.time() - start_time))

# # For timing:
# T = [];
# for t in range(10):
#     start_time = time.time()
#     reinitialise_ReLU_network(model, X, R10_labels)
#     T += [time.time() - start_time]
#     print("--- %s seconds ---" % T[t])
# print(np.mean(np.array(T)))

# For profiling:
reinitialise_ReLU_network(model, X, R10_labels)
