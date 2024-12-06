import numpy as np
from sklearn.model_selection import StratifiedKFold

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
from torch.utils.data import Subset, DataLoader
import git 
import utils.load_data as ld

from src.models.grid_search_CNN import ConvNet
from src.utils import utils

PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(PATH_TO_ROOT)


if __name__ == "__main__":
    config_path = utils.get_config_path(
        default_path=PATH_TO_ROOT + "/src/grid_search/grid_search.yaml"
    )

    config = utils.get_config(config_path)
    batch_size = config["batch_size"]
    kfold = StratifiedKFold(n_splits=config["n_splits"])

    transform = tv.transforms.Compose([
        tv.transforms.ToTensor()
        ])
    train = tv.datasets.MNIST(root=PATH_TO_ROOT+'data/', train=True, download=True, transform=transform)
    test = tv.datasets.MNIST(root=PATH_TO_ROOT+'data', train=False,transform=transform, download=False)

    train_loader = DataLoader(train, batch_size, shuffle=True)

    # Train model
    val_accuracies = []

    for k, (train, val) in enumerate(kfold):
        net = ConvNet(layer_config)
        criterion, optimizer = set_loss(net)