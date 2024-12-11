import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision
import git
import sys

PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(PATH_TO_ROOT)
from src.utils import utils, model_utils
from src.models.CNN import ConvNet

def get_bootstrap_sample(dataset: torch.Tensor, config: dict, seed: int) -> DataLoader:
        batch_size = config["batch_size"]
        N = len(dataset)
        torch.manual_seed(i)
        rng = np.random.default_rng(seed=i)
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
        default_path=PATH_TO_ROOT + "/src/grid_search/CNN/run_cnn_eval"
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
        bootstrap(trainset, config)

        model = model_utils.train_model(trainset)

        PATH = PATH_TO_ROOT+'/saved_models/mnist_net.pth'
        torch.save(model.state_dict(), PATH)

        net = ConvNet()
        net.load_state_dict(torch.load(PATH, weights_only=True))

        # Train final model
        #model_pipeline()

        # Model evaluation
        total_accuracies = bootstrap_test_set()
