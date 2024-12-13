import sys

import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import git

PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(PATH_TO_ROOT)
from src.utils import utils, model_utils, bootstrap
from src.models.CNN import ConvNet


if __name__=="__main__":
        config_path = utils.get_config_path(
        default_path=PATH_TO_ROOT + "/src/evaluation/LogReg/run_logreg_eval.yaml"
        )
        config = utils.get_config(config_path)
        save_plot = config['save_plot']
        torch.manual_seed(config["seed"])
        batch_size = config["batch_size"]
        print_interval = config["print_interval"]

        transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
        ])
        trainset = torchvision.datasets.MNIST(root=PATH_TO_ROOT+'data/', train=True, download=False, transform=transform)
        testset = torchvision.datasets.MNIST(root=PATH_TO_ROOT+'data', train=False,download=False, transform=transform) 

        # Model training 
        trainloader = DataLoader(trainset, batch_size=batch_size)
        model = model_utils.train_model(trainloader, config)

        # Model evaluation
        testloader = DataLoader(testset, batch_size=batch_size)
        accuracy = model_utils.test_model(testloader, model)
        print(f'Accuracy on test set without boostrap is {accuracy}%')

        classes = testset.classes
        score_dict = model_utils.test_model_classwise(testloader, model, classes)
        utils.plot_classwise(score_dict, model="LogReg", save_plot=save_plot)

        # Boostrapped model evaluation
        # total_accuracies = bootstrap.bootstrap_test_set(testset, model, config)
        # lower_bound = np.percentile(total_accuracies, 2.5)
        # upper_bound = np.percentile(total_accuracies, 97.5)
        # mean_accuracy = np.mean(total_accuracies)
        # print(lower_bound, mean_accuracy, upper_bound)
        # fig, ax = plt.subplots()
        # print(len(total_accuracies))
        # sns.histplot(total_accuracies, element="poly", common_norm=False, ax=ax)
        # plt.title("Accuracy on test set")
        # plt.show()


