import torch
import torch.nn as nn
import torch.nn.functional as F


# Create model for MNIST classification (10 classes)
class LogisticRegression(nn.Module):
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
    