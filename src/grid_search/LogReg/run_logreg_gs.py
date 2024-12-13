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
from src.utils import utils
from src.utils.cross_validation import run_cv

"""
Script in progress, need to decide how to grid search logreg before proceeding
"""
if __name__ == "__main__":
    config_path = utils.get_config_path(
        default_path=PATH_TO_ROOT + "/src/grid_search/LogReg/run_logreg_gs.yaml"
    )

    config = utils.get_config(config_path)
    torch.manual_seed(config["seed"])


    transform = tv.transforms.Compose([
        tv.transforms.ToTensor()
    ])
    trainset = tv.datasets.MNIST(root=PATH_TO_ROOT+'/data/', train=True, download=True, transform=transform)
    testset = tv.datasets.MNIST(root=PATH_TO_ROOT+'/data/', train=False,transform=transform, download=False) 


    # Get parameter lists from config
    fixed_lr = config['fixed_lr']
    fixed_batch_size = config['fixed_batch_size']
    fixed_epochs = config['fixed_epochs']
    best_learning_rates = [0.01, 0.1]
    learning_rates = config["learning_rates"]
    batch_sizes = config["batch_sizes"]
    epochs_list = config["epochs_list"]


    # Lists to store results
    lr_accuracies = []
    batch_accuracies = []
    epoch_accuracies = []

    for lr in learning_rates:
        val_accuracies = run_cv(trainset, config, epochs=fixed_epochs, learning_rate=lr, batch_size=fixed_batch_size)
        mean_accuracy = float(np.mean(val_accuracies))
        std_accuracy = float(np.std(val_accuracies))
        lr_accuracies.append(mean_accuracy)
        print(f"Learning Rate: {lr}, CV Accuracy: {mean_accuracy:.2f}% with standard deviation {std_accuracy:.2f}")

    for batch_size in batch_sizes:
        val_accuracies = run_cv(trainset, config, epochs=fixed_epochs, learning_rate=fixed_lr, batch_size=batch_size)
        mean_accuracy = float(np.mean(val_accuracies))
        std_accuracy = float(np.std(val_accuracies))
        batch_accuracies.append(mean_accuracy)
        print(f"Batch size: {batch_size}, CV Accuracy: {mean_accuracy:.2f}% with standard deviation {std_accuracy:.2f}")

    for epoch in epochs_list:
        val_accuracies = run_cv(trainset, config, epochs=epoch, learning_rate=fixed_lr, batch_size=fixed_batch_size)
        mean_accuracy = float(np.mean(val_accuracies))
        std_accuracy = float(np.std(val_accuracies))
        epoch_accuracies.append(mean_accuracy)
        print(f"Epochs: {epoch}, CV Accuracy: {mean_accuracy:.2f}% with standard deviation {std_accuracy:.2f}")


    #Create plots
    utils.plot_parameter_study(learning_rates, lr_accuracies, "Learning Rate")
    utils.plot_parameter_study(batch_sizes, batch_accuracies, "Batch Size", xticks=True)
    utils.plot_parameter_study(epochs_list, epoch_accuracies, "Number of Epochs")







