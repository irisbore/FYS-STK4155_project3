import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvNet(nn.Module):
    def __init__(self, layer_configs):
        """creates a convolutional neural network class

        Args:
            layer_configs (list of dict): each dict is the config of a layer
                'type' = either "conv", "linear"
                #linear
                    'in_features' = input dimensions
                    'out_features' = output dimensions
                #conv
                    'in_channels' = input dimensions for conv
                    'out_channels' = output dimensions for conv
                    'kernel_size' = kernel size for conv
                'activation' = either "sigmoid", "ReLU" or undefined/None
                'pooling' = undefined if none, kernel_size: int if you want pooling
            (including activation function and pooling)
        """
        super(ConvNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layer_configs = layer_configs

        for config in self.layer_configs:
            
            #hidden layer
            if config['type'] == "linear":
                if not any(isinstance(layer, nn.Flatten) for layer in self.layers): # -> size = height x width x channels
                    self.layers.append(nn.Flatten(start_dim=1)) #all execept batch
                self.layers.append(nn.Linear(config['in_features'], config['out_features']))
            elif config['type'] == "conv": #stride = 1, padding = 0 (default)
                inc = config['in_channels']
                outc = config['out_channels']
                k_sz = config['kernel_size']
                self.layers.append(nn.Conv2d(inc, outc, k_sz))
            
            #activation func
            activation = config.get('activation', None)
            if activation:
                    act = self._get_activation(activation) #built in method from pytorch
                    self.layers.append(act) #adds activation as the next "layer"

            #pooling
            pooling = config.get('pooling', None) #stride = 2
            if pooling:
                self.layers.append(nn.MaxPool2d(pooling))

    def _get_activation(self, activation):
        activation = activation.lower()
        if activation == "relu":
            return nn.ReLU()
        elif activation == "leakyrelu":
            return nn.LeakyReLU()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "softmax":
            return nn.Softmax(dim=1) 
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return(x)
    
    def get_flattened_size(self):
        #calculate size after flattening after all convolutional layers
        #conv: default stride = 1, padding = 0
        #output: (input-kernel+padding x 2) / stride + 1
        #pooling: default stride = 2
        #output: (input-kernel)/stride + 1
        #flattening: output is channels x height x width 
        
        size = 28 #28 x 28 that is at the start
        channels = 1 #1 at the start; just records the last output channels
        for config in self.layer_configs:
            if config['type'] == "conv":
                channels = config['out_channels']
                kernel = config['kernel_size']
                padding = config.get('padding', 0) #if we add padding
                stride = config.get('stride', 1)
                size = int((size - kernel + padding*2)/stride + 1) # round it down in case of decimals
            
            pooling = config.get('pooling', None)
            if pooling:
                stride = 2
                kernel = pooling
                size = int((size - kernel)/stride + 1) # round it down in case of decimals
        
        #flatten
        return int(size ** 2 * channels)