import sys
import git
import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler  # for feature scaling
import numpy as np
from sklearn.model_selection import train_test_split  # for train/test split


PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(PATH_TO_ROOT)

from src.utils import utils, load_data
from src.models import LogisticRegression

if __name__ == "__main__":
    config_path = utils.get_config_path(
        default_path=PATH_TO_ROOT + "/src/run_nn.yaml"
    )

    config = utils.get_config(config_path)

    #df = pd.read_csv(PATH_TO_ROOT + config["data_path"])

    rs = np.random.RandomState(config["seed"])

    torch.manual_seed(config["seed"]) 

    # Prepare data
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    n_samples, n_features = X.shape
    print(f'number of samples: {n_samples}, number of features: {n_features}')

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["test_size"], random_state=1234)

    # scale data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # convert to tensors
    X_train = torch.from_numpy(X_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))

    # reshape y tensors
    y_train = y_train.view(y_train.shape[0], 1)
    y_test = y_test.view(y_test.shape[0], 1)

    model = LogisticRegression(n_features)

    # Loss and optimizer
    learning_rate = 0.01
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # training loop
    num_epochs = 100

    for epoch in range(num_epochs):
        # forward pass and loss
        y_predicted = model(X_train)
        loss = criterion(y_predicted, y_train)
        
        # backward pass
        loss.backward()
        
        # updates
        optimizer.step()
        
        # zero gradients
        optimizer.zero_grad()
        
        if (epoch+1) % 10 == 0:
            print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')