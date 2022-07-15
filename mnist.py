# loading MNIST
import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torchvision
#import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
import numpy as np

#train_dataset = torchvision.datasets.MNIST(root='./data',
#                                           train=True,
#                                           download=True)
#train_loader = torch.utils.data.DataLoader(train_dataset,
#                                           batch_size=64,
#                                           shuffle=True)

# show some random training images
# def imshow(img):
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
# dataiter = iter(train_loader)
# images, labels = dataiter.next()
# imshow(torchvision.utils.make_grid(images))


# loading projected MNIST
# MNISTprojected.npy and MNISTlabels.npy created using:
#  X = torch.flatten(train_dataset.data,1)
#  from sklearn.decomposition import PCA
#  pca = PCA(n_components=2)
#  X = pca.fit_transform(X)
# X = np.load("MNISTprojected.npy")
# Xlabels = np.load("MNISTlabels.npy")

# loads N points from each class
def sample_MNIST(train_dataset,N,random=False):
    indices = np.arange(len(train_dataset.targets))
    if random:
        np.random.shuffle(indices)
    label_counter = np.zeros(10)
    Xout = []
    Xlabelsout = []
    for i in indices:
        if label_counter[train_dataset.targets[i]]<N:
            Xout += [torch.unsqueeze(train_dataset.data[i], 0)]
            Xlabelsout += [torch.unsqueeze(train_dataset.targets[i], 0)]
            label_counter[train_dataset.targets[i]] += 1
        if len(Xlabelsout) == N*10:
            break
    XXout = torch.Tensor(len(Xout), 28, 28)
    XXlabelsout = torch.Tensor(len(Xlabelsout), 1)
    torch.cat(Xout, out=XXout)
    torch.cat(Xlabelsout, out=XXlabelsout)
    return np.array(torch.flatten(XXout, 1)), XXlabelsout

def sample_MNIST_projected(X,Xlabels,N,random=False):
    indices = np.arange(len(Xlabels))
    if random:
        np.random.shuffle(indices)
    label_counter = np.zeros(10)
    Xout = []
    Xlabelsout = []
    for i in indices:
        if label_counter[Xlabels[i]]<N:
            Xout += [X[i]]
            Xlabelsout += [Xlabels[i]]
            label_counter[Xlabels[i]] += 1
        if len(Xlabelsout) == N*10:
            break
    return np.array(Xout), np.array(Xlabelsout)

# load_MNIST_projected(X,Xlabels,1)
