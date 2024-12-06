from src.models.grid_search_CNN import ConvNet
import src.utils.load_data as ld
import sys
import git
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#hyperparameters
learning_rate = 10e-3

momentum = 0.9 # for Adam


PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(PATH_TO_ROOT)

torch.manual_seed(1)

def set_loss(model, lr=learning_rate, momentum = momentum):
    pass