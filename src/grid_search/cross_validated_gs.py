import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
from torch.utils.data import Subset, DataLoader
import git 

PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(PATH_TO_ROOT)

from src.models.grid_search_CNN import ConvNet
from src.utils import utils

if __name__ == "__main__":
    config_path = utils.get_config_path(
        default_path=PATH_TO_ROOT + "/src/grid_search/grid_search.yaml"
    )
    config = utils.get_config(config_path)
    torch.manual_seed(config["seed"])
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    epochs = config["epochs"]
    print_interval = config["print_interval"]

    transform = tv.transforms.Compose([
        tv.transforms.ToTensor()
        ])
    trainset = tv.datasets.MNIST(root=PATH_TO_ROOT+'data/', train=True, download=True, transform=transform)
    testset = tv.datasets.MNIST(root=PATH_TO_ROOT+'data', train=False,transform=transform, download=False) 

    cv_accuracy = {}
    for kernel_size in config["kernel_size"]:
        for filter_numbers in config["filter_numbers"]:
            layer_configs = (
                {
                    'type':  "conv",
                    'in_channels': 1,
                    'out_channels': filter_numbers[0],
                    'kernel_size': kernel_size,
                    'activation': "ReLU",
                    'pooling': 2
                },
                {
                    'type':  "conv",
                    'in_channels': filter_numbers[0],
                    'out_channels': filter_numbers[1],
                    'kernel_size': kernel_size,
                    'activation': "ReLU",
                    'pooling': 2
                },
                {
                    'type':  "linear",
                    'in_features': 0,
                    'out_features': 120,
                    'activation': "ReLU",
                },
                {
                    'type':  "linear",
                    'in_features': 120,
                    'out_features': 84,
                    'activation': "ReLU",
                },
                {
                    'type':  "linear",
                    'in_features': 84,
                    'out_features': 10,
                }
            )
            dummynet = ConvNet(layer_configs)
            layer_configs[2]['in_features'] = dummynet.get_flattened_size()
            # Initialize cross validation
            kfold = StratifiedKFold(n_splits=config["n_splits"]).split(trainset, trainset.targets)
            val_accuracies = []
            for k, (train_idx, val_idx) in enumerate(kfold):
                trainloader = DataLoader(trainset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
                valloader = DataLoader(trainset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_idx))

                model = ConvNet(layer_configs)
                criterion, optimizer = utils.set_loss_optim(model, learning_rate)
                for epoch in tqdm(range(epochs)):
                    running_loss = 0.0
                    for i, data in enumerate(trainloader):
                        inputs, labels = data #list of [inputs, labels]
                        optimizer.zero_grad()

                        #forward
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        # print stats
                        running_loss += loss.item()
                        if i % print_interval == print_interval-1: #print every interval
                            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                            running_loss = 0.0

                # Test on whole data set
                correct = 0
                total = 0
                # since we're not training, we don't need to calculate the gradients for our outputs
                with torch.no_grad():
                    for data in valloader:
                        images, labels = data
                        # calculate outputs by running images through the network
                        outputs = model(images)
                        # the class with the highest energy is what we choose as prediction
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                val_accuracy = 100 * correct // total
                val_accuracies.append(val_accuracy)
            cv_accuracy[f'k{kernel_size}_f{filter_numbers}']= np.mean(val_accuracies)
            print(f'kernel size: {kernel_size}, filter number: {filter_numbers}, cv accuracy: {np.mean(val_accuracies)}, cv std: {np.std(val_accuracies)}')
