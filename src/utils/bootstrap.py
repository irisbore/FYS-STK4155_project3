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
        batch_size = config["batch_size"]
        N = len(dataset)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed=seed)
        idx = rng.choice(N, N, replace=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(idx))
        return dataloader

def bootstrap_test_set(testset: torch.Tensor, model, config: dict) -> List:
        total_accuracies = []
        for i in tqdm(range(config["n_bootstraps"])):
                testloader = get_bootstrap_sample(testset, config, i)
                accuracy = model_utils.test_model(testloader, model)
                total_accuracies.append(accuracy)

        return total_accuracies