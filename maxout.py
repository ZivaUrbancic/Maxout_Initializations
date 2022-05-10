import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

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

mySoftmax = torch.nn.Softmax(dim=0)

class MaxoutNet(nn.Module):
    def __init__(self):
        super(MaxoutNet, self).__init__()
        self.lay1lin1 = nn.Linear(28*28, 128)
        self.lay1lin2 = nn.Linear(28*28, 128)
        self.lay1lin3 = nn.Linear(28*28, 128)
        self.lay2lin1 = nn.Linear(128, 10)
        self.lay2lin2 = nn.Linear(128, 10)
        self.lay2lin3 = nn.Linear(128, 10)
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
        # x = mySoftmax(x) # wth does this make loss worse?
        return x

model = MaxoutNet().to(device)

X, X_labels = sample_MNIST(train_dataset, 3000)

#unit_vecs = np.eye(10)
#R10_labels = np.array([unit_vecs[i] for i in X_labels.int()])

reinitialise_Maxout_network(model, X, X_labels.long())

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #print(i)
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 200 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
# PATH = './maxoutMNIST.pth'
# torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(labels.size(0)):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')
        
        

