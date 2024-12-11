import sys
import git
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from models.CNN import ConvNet
import utils.load_data as ld

#hyperparameters
learning_rate = 1e-3
epochs = 2
torch.manual_seed(1)
print_interval = 2000
#potentially add loss calculation and optimizer chooser
layer_configs = (
    {
        'type':  "conv",
        'in_channels': 1,
        'out_channels': 32,
        'kernel_size': 5,
        'activation': "ReLU",
        'pooling': 2 # kernel size
    },
    {
        'type':  "conv",
        'in_channels': 32,
        'out_channels': 64,
        'kernel_size': 5,
        'activation': "ReLU",
        'pooling': 2 # kernel size
    },
    {
        'type':  "linear",
        'in_features': 64*4*4, #1024
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

    #train
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data #list of [inputs, labels]
            optimizer.zero_grad()

            #forward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print stats
            running_loss += loss.item()
            if i % print_interval == print_interval-1: #print every interval
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    
    #test
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = net(inputs)
            #max along columns -> max value, index (only care about index)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(net.get_flattened_size())
    print("Finished Training")
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

