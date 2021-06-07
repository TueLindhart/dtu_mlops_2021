import torch

from torchvision import datasets, transforms


def mnist(batch_size=64):
    # exchange with the real mnist dataset
    #train = torch.randn(50000, 784)
    #test = torch.randn(10000, 784)

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ]) 

    # Download and load the training data
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


    return trainloader, testloader
