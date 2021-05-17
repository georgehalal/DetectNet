import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionNet(nn.Module):
    def __init__(self, params):
        """Define the building blocks of the model
        
        Args:
            params: model hyperparameters
        """

        super(DetectionNet, self).__init__()

        self.cond = nn.Sequential(nn.Linear(6, params.num_nodes), nn.ReLU(True),
            nn.Linear(params.num_nodes, params.num_nodes//2), nn.ReLU(True))

        self.true = nn.Sequential(nn.Linear(4, params.num_nodes), nn.ReLU(True),
            nn.Linear(params.num_nodes, params.num_nodes//2), nn.ReLU(True))

        self.out = nn.Sequential(nn.Linear(params.num_nodes, params.num_nodes),
            nn.ReLU(True),
            nn.Dropout(params.dropout_rate),
            nn.Linear(params.num_nodes, params.num_nodes), nn.ReLU(True),
            nn.Linear(params.num_nodes, 1))
    
    def forward(self, y, t):
        """Define how the model operates on the input batch

        Args:
            y: Observing conditions
            t: Ground truth magnitudes
        """

        y = self.cond(y)
        t = self.true(t)
        x = torch.sigmoid(self.out(torch.cat([y,t],-1)))
        return x


def loss_fn(out, truth):
    """Define loss function to be Binary Cross-Entropy
    
    Args:
        out: model output
        truth: ground truth output
    """

    loss = nn.BCELoss()
    return loss(out, truth)


def accuracy(out, truth):
    """Calculate the accuracy of the prediction

    Args:
        out: model output
        truth: ground truth output
    """

    return np.sum(out==truth)/float(truth.size)
