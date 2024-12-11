
import sys
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim 
from torch.utils.data import DataLoader
import git 
import yaml

PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(PATH_TO_ROOT)

from src.models.CNN import ConvNet
from src.models.LogisticRegression import LogisticRegression
from src.utils import utils, model_utils

def run_cv(trainset: torch.Tensor, config: dict, layer_configs: dict=None, learning_rate: float = None, batch_size = None, epochs:int=None) -> dict:
            # Initialize cross validation
            kfold = StratifiedKFold(n_splits=config["n_splits"]).split(trainset, trainset.targets)
            val_accuracies = []

            if batch_size == None:
                batch_size = config['batch_size']
            for k, (train_idx, val_idx) in enumerate(kfold):
                torch.manual_seed(k)
                trainloader = DataLoader(trainset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
                valloader = DataLoader(trainset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_idx))
                model = model_utils.train_model(trainloader=trainloader, config=config, layer_configs=layer_configs, learning_rate=learning_rate, epochs=epochs)
                val_accuracy = model_utils.test_model(testloader=valloader, model=model)
                val_accuracies.append(val_accuracy)
            return val_accuracies


#######################################################
# def run_cv(trainset: torch.Tensor, config: dict, epochs: int, learning_rate: float, layer_configs: dict=None, batch_size = None) -> dict:
#             # Initialize cross validation
#             kfold = StratifiedKFold(n_splits=config["n_splits"]).split(trainset, trainset.targets)
#             val_accuracies = []

#             if batch_size == None:
#                 batch_size = config['batch_size']
#             for k, (train_idx, val_idx) in enumerate(kfold):
#                 torch.manual_seed(k)

#                 trainloader = DataLoader(trainset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
#                 valloader = DataLoader(trainset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_idx))

#                 #Initialize model with grid search values
#                 if config['model_type'] == "cnn":
#                     model = ConvNet(layer_configs)
#                     criterion = nn.CrossEntropyLoss()
#                     optimizer  = torch.optim.Adam(model.parameters(), lr=learning_rate)
                
#                 if config['model_type'] == "logreg":
#                     model = LogisticRegression()
#                     criterion = nn.NLLLoss()
#                     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
                    
#                 for epoch in tqdm(range(epochs)):
#                     running_loss = 0.0
#                     for i, data in enumerate(trainloader):
#                         inputs, labels = data #list of [inputs, labels]
#                         optimizer.zero_grad()

#                         #forward
#                         outputs = model(inputs)
#                         loss = criterion(outputs, labels)
#                         loss.backward()
#                         optimizer.step()

#                         # print stats
#                         running_loss += loss.item()
#                         if i % config['print_interval'] == config['print_interval']-1: #print every interval
#                             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#                             running_loss = 0.0

#                 # Test on whole data set
#                 correct = 0
#                 total = 0
#                 # since we're not training, we don't need to calculate the gradients for our outputs
#                 with torch.no_grad():
#                     for data in valloader:
#                         images, labels = data
#                         # calculate outputs by running images through the network
#                         outputs = model(images)
#                         # the class with the highest energy is what we choose as prediction
#                         _, predicted = torch.max(outputs.data, 1)
#                         total += labels.size(0)
#                         correct += (predicted == labels).sum().item()

#                 val_accuracy = 100 * correct // total
#                 val_accuracies.append(val_accuracy)
#             return val_accuracies