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

from src.models.CNN import ConvNet
from src.utils import utils
from src.utils.cross_validation import run_cv

if __name__ == "__main__":
    config_path = utils.get_config_path(
        default_path=PATH_TO_ROOT + "/src/grid_search/CNN/run_cnn_gs.yaml"
    )
    config = utils.get_config(config_path)
    torch.manual_seed(config["seed"])
    batch_size = config["batch_size"]
    print_interval = config["print_interval"]
    model_type = config["model_type"]

    transform = tv.transforms.Compose([
        tv.transforms.ToTensor()
        ])
    trainset = tv.datasets.MNIST(root=PATH_TO_ROOT+'data/', train=True, download=False, transform=transform)
    testset = tv.datasets.MNIST(root=PATH_TO_ROOT+'data', train=False,transform=transform, download=False) 

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
                

    if config['grid_search'] == 'epochs+lr':
        cv_accuracy = {
            'Epochs': [],
            'Learning Rate': [],
            'CV Accuracy': [],
            'CV Accuracy Std': []
            }
        filter_number = config["filter_number"]
        kernel_size = config["kernel_size"]
        for epochs in config["epochs"]:
            print(f"On epoch number {epochs}")
            for learning_rate in config["learning_rate"]:
                print(f"With learning rate {learning_rate}")
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
                cv_accuracy['Epochs'].append(epochs)
                cv_accuracy['Learning Rate'].append(learning_rate)
                mean_accuracy = float(np.mean(val_accuracies))
                std_accuracy = float(np.std(val_accuracies))
                cv_accuracy['CV Accuracy'].append(mean_accuracy)
                cv_accuracy['CV Accuracy Std'].append(std_accuracy)

    
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
        # First search
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
                'in_features': 84,
                'out_features': 10,
            }
        )
        dummynet = ConvNet(layer_configs)
        layer_configs[2]['in_features'] = dummynet.get_flattened_size()
        val_accuracies = run_cv(trainset=trainset, config=config, epochs=epochs, learning_rate=learning_rate, layer_configs=layer_configs)
        cv_accuracy['Number of Convolutional Layers'].append(1)
        cv_accuracy['Number of Linear Layers'].append(2)
        mean_accuracy = float(np.mean(val_accuracies))
        std_accuracy = float(np.std(val_accuracies))
        cv_accuracy['CV Accuracy'].append(mean_accuracy)
        cv_accuracy['CV Accuracy Std'].append(std_accuracy)

        #Second search
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
                'in_features': 84,
                'out_features': 10,
            }
        )
        dummynet = ConvNet(layer_configs)
        layer_configs[2]['in_features'] = dummynet.get_flattened_size()
        val_accuracies = run_cv(trainset=trainset, config=config, epochs=epochs, learning_rate=learning_rate, layer_configs=layer_configs)
        cv_accuracy['Number of Convolutional Layers'].append(1)
        cv_accuracy['Number of Linear Layers'].append(2)
        mean_accuracy = float(np.mean(val_accuracies))
        std_accuracy = float(np.std(val_accuracies))
        cv_accuracy['CV Accuracy'].append(mean_accuracy)
        cv_accuracy['CV Accuracy Std'].append(std_accuracy)

         
    
    if config['save_results'] == True:
        with open(PATH_TO_ROOT+f'/results/cnn_grid_search/results_'+config['grid_search']+'.yaml', 'w') as file: 
            file.write(yaml.dump(cv_accuracy))
                   