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
from src.utils import utils

def train_model(trainloader: DataLoader, config: dict, layer_configs=None, learning_rate=None, epochs=None):
        if learning_rate == None:
            learning_rate=config['learning_rate']

        if epochs==None:
            epochs = config['epochs']

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

        return model


def test_model(testloader, model):         # Test on data set
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct // total
    return accuracy

def test_model_classwise(testloader, model, classes):
     # prepare to count predictions for each class
    score_df = {classname: 0 for classname in classes}
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        score_df[classname] = accuracy
    return score_df