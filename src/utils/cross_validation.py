
import sys
from sklearn.model_selection import StratifiedKFold
import torch
import torch.optim 
from torch.utils.data import DataLoader
import git 

PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(PATH_TO_ROOT)

from src.utils import model_utils

def run_cv(trainset: torch.Tensor, config: dict, layer_configs: dict=None, learning_rate: float = None, batch_size = None, epochs:int=None) -> dict:
    """
    Performs k-fold cross-validation on the training set. Varies the seed for each run.

    Args:
        trainset (torch.Tensor): The training dataset.
        config (dict): Configuration dictionary containing parameters like number of folds, batch size and model type.
        layer_configs (dict, optional): Configuration for model layers if training a CNN.
        learning_rate (float, optional): Learning rate for training (default is None, meaning it uses the one provided in the scripts config).
        batch_size (int, optional): Batch size for training and validation (default is None, meaning it uses the one provided in the scripts config).
        epochs (int, optional): Number of epochs for training (default is None, meaning it uses the one provided in the scripts config).

    Returns:
        dict: A dictionary of validation accuracies for each fold.
    """
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
