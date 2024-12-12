
import sys
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim 
from torch.utils.data import DataLoader
import git 
import yaml

PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(PATH_TO_ROOT)

from src.models.CNN import ConvNet
from src.models.LogisticRegression import LogisticRegression
from src.utils import utils, model_utils

def run_cv(trainset: torch.Tensor, config: dict, layer_configs: dict=None, learning_rate: float = None, batch_size = None, epochs:int=None) -> dict:
            # Initialize cross validation
            kfold = StratifiedKFold(n_splits=config["n_splits"]).split(trainset, trainset.targets)
            val_accuracies = []

            if batch_size == None:
                batch_size = config['batch_size']
            for k, (train_idx, val_idx) in enumerate(kfold):
                torch.manual_seed(k)
                trainloader = DataLoader(trainset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
                valloader = DataLoader(trainset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_idx))
                model = model_utils.train_model(trainloader=trainloader, config=config, layer_configs=layer_configs, learning_rate=learning_rate, epochs=epochs)
                val_accuracy = model_utils.test_model(testloader=valloader, model=model)
                val_accuracies.append(val_accuracy)
            return val_accuracies
