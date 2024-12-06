from src.models.grid_search_CNN import ConvNet
import src.utils.load_data as ld
import sys
import git
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#hyperparameters
learning_rate = 10e-3
#potentially add loss calculation and optimizer chooser

conv_config = (
    {
        'type' =  "conv"
        'in_channels' = 1
        'out_channels' = 6
    }
)

ConvNet()

PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(PATH_TO_ROOT)

torch.manual_seed(1)

def set_loss_optim(model, lr=learning_rate):
    criterion = nn.CrossEntropyLoss()
    #optimizer
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    return criterion, optimizer

if __name__ == "__main__":
    net = ConvNet()
