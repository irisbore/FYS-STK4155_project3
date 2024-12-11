import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision
import git
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(PATH_TO_ROOT)
from src.utils import utils, model_utils
from src.models.CNN import ConvNet

def get_bootstrap_sample(dataset: torch.Tensor, config: dict, seed: int) -> DataLoader:
        batch_size = config["batch_size"]
        N = len(dataset)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed=seed)
        idx = rng.choice(N, N, replace=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(idx))
        return dataloader

def bootstrap_test_set(testset, model):
        total_accuracies = []
        for i in tqdm(range(config["n_bootstraps"])):
                testloader = get_bootstrap_sample(testset, config, i)
                accuracy = model_utils.test_model(testloader, model)
                total_accuracies.append(accuracy)

        return total_accuracies


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

        # Model uncertainty
        #bootstrap(trainset, config)

        layer_configs=config['layer_configs']
        dummynet = ConvNet(layer_configs)
        layer_configs[2]['in_features'] = dummynet.get_flattened_size()
        
        # If you have not yet trained final model
        #trainloader = DataLoader(trainset, batch_size=batch_size)
        #model = model_utils.train_model(trainloader, config, layer_configs=layer_configs)

        PATH = PATH_TO_ROOT+'/saved_models/mnist_net.pth'
        #torch.save(model.state_dict(), PATH)

        model = ConvNet(layer_configs)
        model.load_state_dict(torch.load(PATH, weights_only=True))

        # Model evaluation
        testloader = DataLoader(testset, batch_size=batch_size)
        accuracy = model_utils.test_model(testloader, model)
        print(f'Accuracy on test set is {accuracy}%')
        classes = testset.classes
        accuracy = model_utils.test_model_classwise(testloader, model, classes)
        # total_accuracies = bootstrap_test_set(testset, model)
        # lower_bound = np.percentile(total_accuracies, 2.5)
        # upper_bound = np.percentile(total_accuracies, 97.5)
        # mean_accuracy = np.mean(total_accuracies)
        # print(lower_bound, mean_accuracy, upper_bound)
        # fig, ax = plt.subplots()
        # print(len(total_accuracies))
        # sns.histplot(total_accuracies, element="poly", common_norm=False, ax=ax)
        # plt.title("Accuracy on test set")
        # plt.show()

        # classes = testset.classes
        # model_utils.test_model_classwise(testloader, model, classes)

        
