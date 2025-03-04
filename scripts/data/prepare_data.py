import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os

#
# This file will download the data from the MNIST dataset and transform it to use for training.
##

# Transforming
transform = transforms.Compose(
   [transforms.ToTensor()]
)

# Downloading the dataset
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Saving location
os.makedirs('./data/saves', exist_ok=True)

# We save the datasets as pyTorch files
torch.save(trainset, './data/saves/trainset.pt')
torch.save(testset, './data/saves/testset.pt')

print("Finished saving data")