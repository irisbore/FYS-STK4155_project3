import sys

import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import git
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(PATH_TO_ROOT)
from src.utils import utils, model_utils, bootstrap
from src.models.CNN import ConvNet
"""
This script trains and evaluates the validated Convolutional Neural Network (CNN) on the MNIST dataset. 

Steps:
1. Loads configuration settings from a YAML file.
2. Sets up data loading for the MNIST training and test datasets.
3. Initializes and trains a CNN model using the provided configuration.
4. Saves the trained model to a file.
5. Evaluates the model on both the training and test sets, printing accuracy results.
6. Computes and plots class-wise accuracy.
7. Performs bootstrapping to estimate the model's test set accuracy with 95% confidence intervals.
8. Saves the bootstrapped results to a plot file.

The model is saved as 'mnist_net.pth' and the evaluation results are saved as 'CNN_classwise_acc.png' and 'cnn_confidence.png'. Both can be found in the results folder. 
The MNIST data set is downloaded to the data folder.
"""

if __name__=="__main__":
        config_path = utils.get_config_path(
        default_path=PATH_TO_ROOT + "/src/evaluation/CNN/run_cnn_eval.yaml"
        )
        config = utils.get_config(config_path)
        torch.manual_seed(config["seed"])
        batch_size = config["batch_size"]
        print_interval = config["print_interval"]
        save_plot = config['save_plot']
        download_data = config['download_data']
        transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
        ])
        trainset = torchvision.datasets.MNIST(root=PATH_TO_ROOT+'/data/', train=True, download=download_data, transform=transform) 
        testset = torchvision.datasets.MNIST(root=PATH_TO_ROOT+'/data/', train=False,download=download_data, transform=transform) 
        trainloader = DataLoader(trainset, batch_size=batch_size)

        layer_configs=config['layer_configs']
        dummynet = ConvNet(layer_configs)
        layer_configs[2]['in_features'] = dummynet.get_flattened_size()
        
        PATH = PATH_TO_ROOT+'/results/saved_models/mnist_net.pth'

        # If you already run this script once you can comment out the following
        #-------------------------------------------------------------------------------------------------------#
        model = model_utils.train_model(trainloader, config, layer_configs=layer_configs)
        torch.save(model.state_dict(), PATH)
        #-------------------------------------------------------------------------------------------------------#
        model = ConvNet(layer_configs)
        model.load_state_dict(torch.load(PATH, weights_only=True))

        # Model evaluation
        # For reference, print training accuracy
        accuracy = model_utils.test_model(trainloader, model)
        print(f'Accuracy on training set is {accuracy}%')

        # Accuracy on test set
        testloader = DataLoader(testset, batch_size=batch_size)
        accuracy = model_utils.test_model(testloader, model)
        print(f'Accuracy on test set is {accuracy}%')
        classes = testset.classes

        # Evaluate classwise accuracy
        score_dict = model_utils.test_model_classwise(testloader, model, classes)
        utils.plot_classwise(score_dict=score_dict, model="CNN", save_plot=True)

        # Boostrapped model evaluation
        total_accuracies = bootstrap.bootstrap_test_set(testset, model, config)
        lower_bound = float(np.percentile(total_accuracies, 2.5))
        upper_bound = float(np.percentile(total_accuracies, 97.5))
        mean_accuracy = float(np.mean(total_accuracies))
        fig, ax = plt.subplots()
        sns.histplot(total_accuracies, element="poly", common_norm=False, ax=ax)
        plt.title(f"Accuracy on test set, with 95% CI: [{lower_bound:.2f}, {upper_bound:.2f}] ")
        plt.savefig(f'{PATH_TO_ROOT}/results/evaluation/cnn_confidence.png')


        
