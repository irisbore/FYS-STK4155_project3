import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision
import git
import sys

from src.utils import utils
PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(PATH_TO_ROOT)

from models.CNN import ConvNet

def bootstrap(dataset: torch.Tensor, config):
        for i in tqdm(range(config["n_bootstraps"])):
                # Generating random indices for sampling from dataframe
                batch_size = config["batch_size"]
                N = len(dataset)
                torch.manual_seed(i)
                rng = np.random.default_rng(seed=i)
                idx = rng.choice(N, N, replace=True)
                dataloader = DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(idx))





if __name__=="__main__":
        config_path = utils.get_config_path(
        default_path=PATH_TO_ROOT + "/src/grid_search/CNN/grid_search.yaml"
        )
        config = utils.get_config(config_path)
        torch.manual_seed(config["seed"])
        batch_size = config["batch_size"]
        print_interval = config["print_interval"]

        transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
        ])
        trainset = torchvision.datasets.MNIST(root=PATH_TO_ROOT+'data/', train=True, download=False, transform=transform)
        testset = torchvision.datasets.MNIST(root=PATH_TO_ROOT+'data', train=False,transform=transform, download=False) 
        full_data = torch.utils.data.ConcatDataset([trainset, testset])
