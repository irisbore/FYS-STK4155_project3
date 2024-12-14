import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticRegression(nn.Module):
    """
    Logistic Regression model for digit classification (e.g., MNIST).

    Args:
        n_input_features (int): Number of input features (default is 784 for 28x28 images).

    Methods:
        forward(x):
            Defines the forward pass of the model. 
            Takes an input tensor, flattens it, computes the logits, and applies softmax.
    """
    def __init__(self, n_input_features=784):  # 28x28 = 784 pixels for MNIST
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 10)  # 10 classes for digits 0-9

    def forward(self, x):
        # Flatten the input image (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.view(x.size(0), -1)
        # Get logits
        logits = self.linear(x)
        # Apply softmax to get probabilities
        return F.log_softmax(logits, dim=1)
    