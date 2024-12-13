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


if __name__=="__main__":
        config_path = utils.get_config_path(
        default_path=PATH_TO_ROOT + "/src/evaluation/CNN/run_cnn_eval.yaml"
        )
        config = utils.get_config(config_path)
        torch.manual_seed(config["seed"])
        batch_size = config["batch_size"]
        print_interval = config["print_interval"]

        transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
        ])
        trainset = torchvision.datasets.MNIST(root=PATH_TO_ROOT+'data/', train=True, download=False, transform=transform)
        testset = torchvision.datasets.MNIST(root=PATH_TO_ROOT+'data', train=False,download=False, transform=transform) 

        layer_configs=config['layer_configs']
        dummynet = ConvNet(layer_configs)
        layer_configs[2]['in_features'] = dummynet.get_flattened_size()
        
        # If you have not yet trained final model
        #trainloader = DataLoader(trainset, batch_size=batch_size)
        #model = model_utils.train_model(trainloader, config, layer_configs=layer_configs)

        PATH = PATH_TO_ROOT+'/results/saved_models/mnist_net.pth'
        #torch.save(model.state_dict(), PATH)

        model = ConvNet(layer_configs)
        model.load_state_dict(torch.load(PATH, weights_only=True))

        # Model evaluation
        testloader = DataLoader(testset, batch_size=batch_size)
        accuracy = model_utils.test_model(testloader, model)
        print(f'Accuracy on test set is {accuracy}%')
        classes = testset.classes

        ## TODO : run this below 
        score_dict = model_utils.test_model_classwise(testloader, model, classes)
        utils.plot_classwise(score_dict=score_dict, model="CNN", save_plot=True)

        #print(score_df)
        #utils.plot_classwise(score_df)

        # Boostrapped model evaluation
        # total_accuracies = bootstrap.bootstrap_test_set(testset, model, config)
        # lower_bound = np.percentile(total_accuracies, 2.5)
        # upper_bound = np.percentile(total_accuracies, 97.5)
        # mean_accuracy = np.mean(total_accuracies)
        # print(lower_bound, mean_accuracy, upper_bound)
        # fig, ax = plt.subplots()
        # print(len(total_accuracies))
        # sns.histplot(total_accuracies, element="poly", common_norm=False, ax=ax)
        # plt.title(f"Accuracy on test set, with 95% CI: [{lower_bound, mean_accuracy, upper_bound}] ")
        # plt.show()


        
