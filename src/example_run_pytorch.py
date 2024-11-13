import sys
import git
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(PATH_TO_ROOT)

from src.utils import CNN as cnn
from src.utils import utils, load_data

"""
Template to show how you can load from config and use the existing CNN class. 
WARNING: The existing class is not a CNN yet!!! It needs to be rewritten, it is just a possible template"""

if __name__ == "__main__":
    config_path = utils.get_config_path(
        default_path=PATH_TO_ROOT + "/src/run_nn.yaml"
    )

    config = utils.get_config(config_path)

    #df = pd.read_csv(PATH_TO_ROOT + config["data_path"])

    rs = np.random.RandomState(config["seed"])

    torch.manual_seed(config["seed"]) 

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["test_size"], random_state=rs)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    input_size = X_train.shape[1]

    # Final model based on grid search
    node_size = 41
    num_hidden_layers = 1
    activation_func = "leaky_relu"
    model = cnn.CNN(input_size, node_size, num_hidden_layers, activation_func)
    criterion = nn.BCELoss()  # Binary Cross-Entropy loss
    learning_rate = 0.1
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            predicted = (test_outputs > 0.5).float()
            accuracy = (predicted.eq(y_test_tensor).sum().item() / y_test_tensor.size(0)) * 100
