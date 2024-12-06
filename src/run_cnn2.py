import sys
import git
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from models.grid_search_CNN import ConvNet
import utils.load_data as ld

#hyperparameters
learning_rate = 10e-3
epochs = 2
torch.manual_seed(1)
#potentially add loss calculation and optimizer chooser
layer_configs = (
    {
        'type':  "conv",
        'in_channels': 1,
        'out_channels': 6,
        'kernel_size': 5,
        'activation': "ReLU",
        'pooling': 2
    },
    {
        'type':  "conv",
        'in_channels': 6,
        'out_channels': 16,
        'kernel_size': 5,
        'activation': "ReLU",
        'pooling': 2
    },
    {
        'type':  "linear",
        'in_features': 16*4*4, #256
        'out_features': 120,
        'activation': "ReLU",
    },
    {
        'type':  "linear",
        'in_features': 120,
        'out_features': 84,
        'activation': "ReLU",
    },
    {
        'type':  "linear",
        'in_features': 84,
        'out_features': 10,
    }
)

PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(PATH_TO_ROOT)

def set_loss_optim(model, lr=learning_rate):
    criterion = nn.CrossEntropyLoss()
    #optimizer
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    return criterion, optimizer

if __name__ == "__main__":
    net = ConvNet(layer_configs=layer_configs)
    trainset, testset, trainloader, testloader = ld.load_transform_MNIST()
    classes = trainset.classes
    print("Classes: ", classes)

    criterion, optimizer = set_loss_optim(net)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data #list of [inputs, labels]
            optimizer.zero_grad

            #forward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()