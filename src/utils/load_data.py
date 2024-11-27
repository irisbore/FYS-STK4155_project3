import sys 

import numpy as np
import git
import matplotlib.pyplot as plt
import sklearn.datasets
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms

PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(PATH_TO_ROOT)

#import toy dataset
def toy_data(n_samples: int, random_state: int):
    pass

#import CMU face dataset - will remove this
def load_as_png(image_path: str):
    im = Image.open("data/dataset/cmu+face+images/faces/"+image_path)
    im.show()

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def load_transform_MNIST(batch_size:int=4):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))]
        )
    
    # Return a tensor for training
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    # Return a tensor for training
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    return trainset, testset, trainloader, testloader

if __name__ == "__main__":
    # get some random training images
    _, _, trainloader, testloader = load_transform_MNIST()
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images))

