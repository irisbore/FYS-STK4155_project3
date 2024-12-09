import sys
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
from torch.utils.data import Subset, DataLoader
import git 
import yaml

PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(PATH_TO_ROOT)

from models.CNN import ConvNet
from src.utils import utils
from src.utils.cross_validation import run_cv

"""
Script in progress, need to decide how to grid search logreg before proceeding
"""
if __name__ == "__main__":
    config_path = utils.get_config_path(
        default_path=PATH_TO_ROOT + "/src/run_LogReg.yaml"
    )

    config = utils.get_config(config_path)
    torch.manual_seed(config["seed"])

    # Get parameter lists from config
    learning_rates = config["learning_rates"]
    batch_sizes = config["batch_sizes"]
    epochs_list = config["epochs_list"]
