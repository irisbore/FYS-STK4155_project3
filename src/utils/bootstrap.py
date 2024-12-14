from typing import List
import sys

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import git


PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(PATH_TO_ROOT)
from src.utils import model_utils


def get_bootstrap_sample(dataset: torch.Tensor, config: dict, seed: int) -> DataLoader:
        """
        Creates a bootstrap sample from the dataset.

        Args:
                dataset (torch.Tensor): The dataset to sample from.
                config (dict): Configuration dictionary containing the batch size.
                seed (int): Random seed for reproducibility.

        Returns:
                DataLoader: DataLoader with bootstrap samples.
        """
        batch_size = config["batch_size"]
        N = len(dataset)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed=seed)
        idx = rng.choice(N, N, replace=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(idx))
        return dataloader

def bootstrap_test_set(testset: torch.Tensor, model, config: dict) -> List:
        """
        Performs bootstrap sampling on the data set and computes accuracies. Varies the seed for each bootstrap run.

        Args:
                testset (torch.Tensor): The test dataset, this method is only intented for test/validation sets.
                model: The model to evaluate.
                config (dict): Configuration dictionary containing the number of bootstraps.

        Returns:
                List: List of accuracies from each bootstrap sample.
        """
        total_accuracies = []
        for i in tqdm(range(config["n_bootstraps"])):
                testloader = get_bootstrap_sample(testset, config, i)
                accuracy = model_utils.test_model(testloader, model)
                total_accuracies.append(accuracy)

        return total_accuracies