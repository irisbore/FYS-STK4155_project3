import sys
import time 
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

"""
This script performs a grid search to tune hyperparameters for training a Convolutional Neural Network (CNN) on the MNIST dataset.

It supports the following grid search configurations:
1. Number of convolutional layers
2. Kernel size and filter number
3. Padding vs. pooling
4. Dropout rate and activation function

The script evaluates each configuration using cross-validation and reports the mean and standard deviation of validation accuracy. 
Results are saved to a YAML file in the results folder if specified in the configuration.

Usage:
    python run_cnn_gs.py --config_path <config_file.yaml>
    Example: python run_cnn_gs.py --config_path run_cnn_gs_conv_layers.yaml
"""


PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(PATH_TO_ROOT)

from src.models.CNN import ConvNet
from src.utils import utils
from src.utils.cross_validation import run_cv

if __name__ == "__main__":
    config_path = utils.get_config_path()
    config = utils.get_config(config_path)
    torch.manual_seed(config["seed"])
    batch_size = config["batch_size"]
    print_interval = config["print_interval"]
    model_type = config["model_type"]

    transform = tv.transforms.Compose([
        tv.transforms.ToTensor()
        ])
    trainset = tv.datasets.MNIST(root=PATH_TO_ROOT+'/data/', train=True, download=True, transform=transform) #CHANGE TO DOWNLOAD=FALSE IF PREVIOUSLY RUN
    testset = tv.datasets.MNIST(root=PATH_TO_ROOT+'/data/', train=False, download=True, transform=transform) 
    
    #---------------------------------------------------------------------------------------------------------------------------------------------#
    # FIRST GRID SEARCH 
    if config["grid_search"] == 'number_of_conv_layers':
        cv_accuracy = {
        'Number of Convolutional Layers': [],
        'Number of Linear Layers' : [],
        'CV Accuracy': [],
        'CV Accuracy Std': []
        }
        learning_rate = config["learning_rate"]
        epochs = config["epochs"]
        filter_number = config["filter_number"]
        kernel_size = config["kernel_size"]
        # First model type
        layer_configs = (
            {
                'type':  "conv",
                'in_channels': 1,
                'out_channels': filter_number[0],
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
                'out_features': 10,
            }
        )
        dummynet = ConvNet(layer_configs)
        layer_configs[1]['in_features'] = dummynet.get_flattened_size()
        val_accuracies = run_cv(trainset=trainset, config=config, epochs=epochs, learning_rate=learning_rate, layer_configs=layer_configs)
        cv_accuracy['Number of Convolutional Layers'].append(1)
        cv_accuracy['Number of Linear Layers'].append(2)
        mean_accuracy = float(np.mean(val_accuracies))
        std_accuracy = float(np.std(val_accuracies))
        cv_accuracy['CV Accuracy'].append(mean_accuracy)
        cv_accuracy['CV Accuracy Std'].append(std_accuracy)

        #Second model type
        layer_configs = (
            {
                'type':  "conv",
                'in_channels': 1,
                'out_channels': filter_number[0],
                'kernel_size': kernel_size,
                'activation': "ReLU",
                'pooling': 2
            },
            {
                'type':  "conv",
                'in_channels': filter_number[0],
                'out_channels': filter_number[1],
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
        val_accuracies = run_cv(trainset=trainset, config=config, epochs=epochs, learning_rate=learning_rate, layer_configs=layer_configs)
        cv_accuracy['Number of Convolutional Layers'].append(2)
        cv_accuracy['Number of Linear Layers'].append(3)
        mean_accuracy = float(np.mean(val_accuracies))
        std_accuracy = float(np.std(val_accuracies))
        cv_accuracy['CV Accuracy'].append(mean_accuracy)
        cv_accuracy['CV Accuracy Std'].append(std_accuracy)

    #---------------------------------------------------------------------------------------------------------------------------------------------#
    # SECOND GRID SEARCH
    if config["grid_search"] == "kernel+filter":
        cv_accuracy = {
        'Kernel Size': [],
        'Filter Number': [],
        'CV Accuracy': [],
        'CV Accuracy Std': []
        }
        learning_rate = config["learning_rate"]
        epochs = config["epochs"]
        for kernel_size in config["kernel_size"]:
            for filter_number in config["filter_number"]:
                layer_configs = (
                    {
                        'type':  "conv",
                        'in_channels': 1,
                        'out_channels': filter_number[0],
                        'kernel_size': kernel_size,
                        'activation': "ReLU",
                        'pooling': 2
                    },
                    {
                        'type':  "conv",
                        'in_channels': filter_number[0],
                        'out_channels': filter_number[1],
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
                val_accuracies = run_cv(trainset=trainset, config=config, epochs=epochs, learning_rate=learning_rate, layer_configs=layer_configs)
                cv_accuracy['Kernel Size'].append(kernel_size)
                cv_accuracy['Filter Number'].append(filter_number)
                mean_accuracy = float(np.mean(val_accuracies))
                std_accuracy = float(np.std(val_accuracies))
                cv_accuracy['CV Accuracy'].append(mean_accuracy)
                cv_accuracy['CV Accuracy Std'].append(std_accuracy)

    #---------------------------------------------------------------------------------------------------------------------------------------------#
    # THIRD GRID SEARCH
    if config["grid_search"] == 'padding_vs_pooling':
        learning_rate = config["learning_rate"]
        epochs = config["epochs"]
        filter_number = config["filter_number"]
        kernel_size = config["kernel_size"]
        padding_list = [0, int((kernel_size-1)/2)]
        cv_accuracy = {
        'Padding': [],
        'Pooling' : [],
        'CV Accuracy': [],
        'CV Accuracy Std': []
        }
        for padding in padding_list:
            for pooling in config['pooling']:
                start_time = time.perf_counter()
                layer_configs = [
                {
                    'type':  "conv",
                    'in_channels': 1,
                    'out_channels': filter_number[0],
                    'kernel_size': kernel_size,
                    'activation': "ReLU",
                    'pooling': pooling,
                    'padding': padding
                },
                {
                    'type':  "conv",
                    'in_channels': filter_number[0],
                    'out_channels': filter_number[1],
                    'kernel_size': kernel_size,
                    'activation': "ReLU",
                    'pooling': pooling,
                    'padding': padding
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
                ]
                dummynet = ConvNet(layer_configs)
                layer_configs[2]['in_features'] = dummynet.get_flattened_size()
                val_accuracies = run_cv(trainset=trainset, config=config, epochs=epochs, learning_rate=learning_rate, layer_configs=layer_configs)
                mean_accuracy = float(np.mean(val_accuracies))
                std_accuracy = float(np.std(val_accuracies))
                cv_accuracy['Padding'].append(padding)
                cv_accuracy['Pooling'].append(pooling)
                cv_accuracy['CV Accuracy'].append(mean_accuracy)
                cv_accuracy['CV Accuracy Std'].append(std_accuracy)
                end_time = time.perf_counter()
                used_time = end_time - start_time
                minutes = used_time // 60
                seconds = used_time % 60
                print(f"Running time for padding: {padding} and pooling: {pooling} is:{int(minutes)} minutes {seconds:.2f} seconds")


    #---------------------------------------------------------------------------------------------------------------------------------------------#
    # FOURTH GRID SEARCH: 
    if config["grid_search"] == 'dropout_vs_activations':
        learning_rate = config["learning_rate"]
        epochs = config["epochs"]
        filter_number = config["filter_number"]
        kernel_size = config["kernel_size"]
        pooling = config["pooling"]
        cv_accuracy = {
        'Dropout Rate': [], 
        'Activation Function' : [],
        'CV Accuracy': [],
        'CV Accuracy Std': []
        }
        for dropout_rate in config['dropout_rate']:
            for activation_function in config['activation_function']:
                layer_configs = [
                {
                    'type':  "conv",
                    'in_channels': 1,
                    'out_channels': filter_number[0],
                    'kernel_size': kernel_size,
                    'activation': activation_function,
                    'pooling': pooling
                },
                {
                    'type':  "conv",
                    'in_channels': filter_number[0],
                    'out_channels': filter_number[1],
                    'kernel_size': kernel_size,
                    'activation': activation_function,
                    'pooling': pooling
                },
                {
                    'type':  "linear",
                    'in_features': 0,
                    'out_features': 120,
                    'activation': activation_function,
                    'dropout': dropout_rate
                },
                {
                    'type':  "linear",
                    'in_features': 120,
                    'out_features': 84,
                    'activation': activation_function,
                    'dropout': dropout_rate
                },
                {
                    'type':  "linear",
                    'in_features': 84,
                    'out_features': 10,
                }
                ]
                dummynet = ConvNet(layer_configs)
                layer_configs[2]['in_features'] = dummynet.get_flattened_size()
                val_accuracies = run_cv(trainset=trainset, config=config, epochs=epochs, learning_rate=learning_rate, layer_configs=layer_configs)
                mean_accuracy = float(np.mean(val_accuracies))
                std_accuracy = float(np.std(val_accuracies))
                cv_accuracy['Dropout Rate'].append(dropout_rate)
                cv_accuracy['Activation Function'].append(activation_function)
                cv_accuracy['CV Accuracy'].append(mean_accuracy)
                cv_accuracy['CV Accuracy Std'].append(std_accuracy)

#---------------------------------------------------------------------------------------------------------------------------------------------#
    if config['save_results'] == True:
        with open(PATH_TO_ROOT+f'/results/cnn_grid_search/results_'+config['grid_search']+'.yaml', 'w') as file: 
            file.write(yaml.dump(cv_accuracy))
                   