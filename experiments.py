###
# Functions for running experiments
###
def load_data(dataset):
    if dataset == "MNIST" or dataset == "mnist":
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
        input_dimension = 28*28
        output_dimension = 10
    if dataset == "CIFAR10" or dataset == "cifar10":
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
    return train_dataset, test_dataset, input_dimension, output_dimension


class ReLUNet3(nn.Module):
    def __init__(self,n0,n1,n2,n3):
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

class ReLUNet2(nn.Module):
    def __init__(self,n0,n1,n2):
        super(ReLUNet, self).__init__()
        self.layer1 = nn.Linear(n0, n1)
        self.act1 = activation
        self.layer2 = nn.Linear(n1, n2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        return output_normalisation(x)


###
# Deepcopy hack
###
def deepcopy_network(model, random_name="hunter2"):
    torch.save(model,random_name+'.pt')
    return torch.load(random_name+'.pt')


def load_optimiser(model,optimiser_name):
    if optimiser_name=="sgd":
        return torch.optim.SGD(model.parameters(),lr=0.001)
    if optimiser_name=="adam":
        return torch.optim.Adam(model.parameters())
