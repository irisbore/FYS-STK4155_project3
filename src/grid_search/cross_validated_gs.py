import numpy as np
from sklearn.model_selection import StratifiedKFold
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import git 
import utils.load_data as ld
from src.models.CNN import Net

PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(PATH_TO_ROOT)

trainset, _, trainloader, _ = ld.load_transform_MNIST()
X_train = trainset.data
y_train = trainset.classes
kfold = StratifiedKFold(n_splits=10).split(trainset, y)

for k, (train, val) in enumerate(kfold):
    pass