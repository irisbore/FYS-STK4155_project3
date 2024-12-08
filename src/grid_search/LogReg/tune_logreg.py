import sys
import git
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(PATH_TO_ROOT)

from src.utils import utils, load_data
from src.models.LogisticRegression import LogisticRegression

"""
This file should be modified to use cross validated accuracy. Data can be loaded as in cnn/cross_val_gs.py, I think we can drop the load_data.py file
"""

def train_and_evaluate(model, trainloader, testloader, criterion, optimizer, n_epochs):
    # Training loop
    n_total_steps = len(trainloader)
    
    for epoch in range(n_epochs):
        model.train()
        for i, (images, labels) in enumerate(trainloader):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
    return acc

def plot_parameter_study(param_values, accuracies, param_name):
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, accuracies, 'bo-')
    plt.xlabel(param_name)
    plt.ylabel('Accuracy (%)')
    plt.title(f'Impact of {param_name} on Model Accuracy')
    plt.grid(True)
    plt.savefig(f'results/{param_name.lower().replace(" ", "_")}_study.png')
    plt.close()

if __name__ == "__main__":
    config_path = utils.get_config_path(
        default_path=PATH_TO_ROOT + "/src/run_LogReg.yaml"
    )

    config = utils.get_config(config_path)
    torch.manual_seed(config["seed"])

    # Get parameter lists from config
    learning_rates = config["learning_rates"]
    batch_sizes = config["batch_sizes"]
    epochs_list = config["epochs_list"]

    # Lists to store results
    lr_accuracies = []
    batch_accuracies = []
    epoch_accuracies = []

    # 1. Learning Rate Study
    print("\nPerforming Learning Rate Study...")
    for lr in learning_rates:
        _, _, trainloader, testloader = load_data.load_transform_MNIST(batch_size=64)
        model = LogisticRegression()
        criterion = nn.NLLLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        
        acc = train_and_evaluate(model, trainloader, testloader, criterion, optimizer, n_epochs=10)
        lr_accuracies.append(acc)
        print(f"Learning Rate: {lr}, Accuracy: {acc:.2f}%")

    # 2. Batch Size Study
    print("\nPerforming Batch Size Study...")
    for bs in batch_sizes:
        _, _, trainloader, testloader = load_data.load_transform_MNIST(batch_size=bs)
        model = LogisticRegression()
        criterion = nn.NLLLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        acc = train_and_evaluate(model, trainloader, testloader, criterion, optimizer, n_epochs=10)
        batch_accuracies.append(acc)
        print(f"Batch Size: {bs}, Accuracy: {acc:.2f}%")

    # 3. Epochs Study
    print("\nPerforming Epochs Study...")
    _, _, trainloader, testloader = load_data.load_transform_MNIST(batch_size=64)
    for epochs in epochs_list:
        model = LogisticRegression()
        criterion = nn.NLLLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        acc = train_and_evaluate(model, trainloader, testloader, criterion, optimizer, n_epochs=epochs)
        epoch_accuracies.append(acc)
        print(f"Epochs: {epochs}, Accuracy: {acc:.2f}%")

    # Create plots
    plot_parameter_study(learning_rates, lr_accuracies, "Learning Rate")
    plot_parameter_study(batch_sizes, batch_accuracies, "Batch Size")
    plot_parameter_study(epochs_list, epoch_accuracies, "Number of Epochs")

    print("\nParameter study completed. Plots have been saved.")