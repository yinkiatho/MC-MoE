import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32], output_dim: int = 1, dropout: float = 0.0):
        """
        Args:
            input_dim: number of input features
            hidden_dims: list of hidden layer sizes, e.g. [64, 32]
            output_dim: number of output units
            dropout: dropout rate between layers
        """
        super(MLPModel, self).__init__()

        self.layers = nn.ModuleList()
        prev_dim = input_dim

        # Create hidden layers
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim

        self.output_layer = nn.Linear(prev_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        # Pass through hidden layers with ReLU
        for layer in self.layers:
            x = F.relu(layer(x))
            if self.dropout.p > 0:
                x = self.dropout(x)
                
        # Output layer (no activation, can add sigmoid/softmax outside if needed)
        x = self.output_layer(x)
        return x