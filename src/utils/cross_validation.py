
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
from src.models.LogisticRegression import LogisticRegression
from src.utils import utils

def run_cv(trainset: torch.Tensor, config_path: str, epochs: int, learning_rate: float, layer_configs: dict=None) -> dict:
            config = utils.get_config(config_path)

            # Initialize cross validation
            kfold = StratifiedKFold(n_splits=config["n_splits"]).split(trainset, trainset.targets)
            val_accuracies = []
            for k, (train_idx, val_idx) in enumerate(kfold):
                torch.manual_seed(k)

                trainloader = DataLoader(trainset, batch_size=config['batch_size'], sampler=torch.utils.data.SubsetRandomSampler(train_idx))
                valloader = DataLoader(trainset, batch_size=config['batch_size'], sampler=torch.utils.data.SubsetRandomSampler(val_idx))

                #Initialize model with grid search values
                if config['model_type'] == "cnn":
                    model = ConvNet(layer_configs)
                    criterion = nn.CrossEntropyLoss()
                    optimizer  = torch.optim.Adam(model.parameters(), lr=learning_rate)
                
                if config['model_type'] == "logreg":
                    model = LogisticRegression()
                    criterion = nn.NLLLoss()
                    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
                    
                for epoch in tqdm(range(epochs)):
                    running_loss = 0.0
                    for i, data in enumerate(trainloader):
                        inputs, labels = data #list of [inputs, labels]
                        optimizer.zero_grad()

                        #forward
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        # print stats
                        running_loss += loss.item()
                        if i % config['print_interval'] == config['print_interval']-1: #print every interval
                            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                            running_loss = 0.0

                # Test on whole data set
                correct = 0
                total = 0
                # since we're not training, we don't need to calculate the gradients for our outputs
                with torch.no_grad():
                    for data in valloader:
                        images, labels = data
                        # calculate outputs by running images through the network
                        outputs = model(images)
                        # the class with the highest energy is what we choose as prediction
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                val_accuracy = 100 * correct // total
                val_accuracies.append(val_accuracy)
            return val_accuracies