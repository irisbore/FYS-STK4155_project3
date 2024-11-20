import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(2002) 

class CNN(nn.Module):
    """
    A feedforward neural network model for binary classification

    Attributes:
        output (nn.Linear): Output layer, producing a single value (binary classification).
        layers (nn.ModuleList): List of hidden layers, each a fully connected layer.
        activation_func (nn.Module): Activation function applied to the hidden layers.
    
    Args:
        input_size (int): The number of input features.
        node_size (int): The number of neurons in each hidden layer.
        num_hidden_layers (int): The number of hidden layers in the network.
        activation (str): The activation function to use in hidden layers. 
                          Options are 'relu', 'leaky_relu', 'sigmoid'. 
    
    Raises:
        ValueError: If the provided activation function is not one of the recognized types ('relu', 'leaky_relu', 'sigmoid').

    Methods:
        forward(x):
            Passes input through the network, returning the predicted output.
    """
    def __init__(self, input_size, node_size, num_hidden_layers, activation):
        super(CNN, self).__init__()

        self.output = nn.Linear(node_size, 1) #output
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, node_size))  #Input
        self.activation_func = None 

        #nn.Conv2d
        if activation == 'relu':
            self.activation_func = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation_func = nn.LeakyReLU()
        elif activation == 'sigmoid':
            self.activation_func = nn.Sigmoid()
        else:
            raise ValueError

        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(node_size, node_size))

    def forward(self, x):
        for layer in self.layers:
            x = self.activation_func(layer(x))

        return torch.sigmoid(self.output(x))
        