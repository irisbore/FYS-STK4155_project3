import sys
import git
import torch
import torch.nn as nn
import numpy as np


PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(PATH_TO_ROOT)

from src.utils import utils, load_data
from src.models.LogisticRegression import LogisticRegression

if __name__ == "__main__":
    config_path = utils.get_config_path(
        default_path=PATH_TO_ROOT + "/src/run_LogReg.yaml"
    )

    config = utils.get_config(config_path)
    torch.manual_seed(config["seed"])

    # Load MNIST data
    batch_size = 64
    _, _, trainloader, testloader = load_data.load_transform_MNIST(batch_size=batch_size)

    # Create model
    model = LogisticRegression()
    
    # Loss and optimizer
    criterion = nn.NLLLoss()  # Negative Log Likelihood Loss for log_softmax output
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    n_epochs = 10
    n_total_steps = len(trainloader)

    for epoch in range(n_epochs):
        model.train()
        for i, (images, labels) in enumerate(trainloader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    print('Finished Training')

    # Test the model
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
        print(f'Accuracy of the model on the test images: {acc} %')