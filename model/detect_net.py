# -*- coding: utf-8 -*-
"""
detect_net.py

Contains the model, the loss function, and the accuracy function.

Author: George Halal
Email: halalgeorge@gmail.com
"""


__author__ = "George Halal"
__email__ = "halalgeorge@gmail.com"


import sys
sys.path.append("../")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Params


class DetectionNet(nn.Module):
    """Model for the detection probability of galaxy detection in wide-field
    surveys.
    """

    def __init__(self, params: Params) -> None:
        """Define the building blocks of the model
        
        Args:
            params (Params): model hyperparameters
        """
        super(DetectionNet, self).__init__()

        self.cond = nn.Sequential(
            nn.Linear(6, params.num_nodes), nn.ReLU(True),
            nn.Linear(params.num_nodes, params.num_nodes // 2), nn.ReLU(True))

        self.true = nn.Sequential(
            nn.Linear(4, params.num_nodes), nn.ReLU(True),
            nn.Linear(params.num_nodes, params.num_nodes // 2), nn.ReLU(True))

        self.out = nn.Sequential(
            nn.Linear(params.num_nodes, params.num_nodes), nn.ReLU(True),
            nn.Dropout(params.dropout_rate),
            nn.Linear(params.num_nodes, params.num_nodes), nn.ReLU(True),
            nn.Linear(params.num_nodes, 1))
    
        return None

    def forward(self, y: torch.tensor, t: torch.tensor) -> torch.tensor:
        """Define how the model operates on the input batch

        Args:
            y (torch.tensor): Observing conditions
            t (torch.tensor): Ground truth magnitudes

        Returns:
            x (torch.tensor): Detection probability
        """
        y = self.cond(y)
        t = self.true(t)
        x = torch.sigmoid(self.out(torch.cat([y, t], -1)))

        return x


def loss_fn(out: torch.tensor, truth: torch.tensor) -> torch.tensor:
    """Define loss function to be Binary Cross-Entropy
    
    Args:
        out (torch.tensor): model output
        truth (torch.tensor): ground truth output

    Returns:
        (torch.tensor): binary cross-entropy loss
    """
    loss = nn.BCELoss()

    return loss(out, truth)


def accuracy(out: torch.tensor, truth: torch.tensor) -> float:
    """Calculate the accuracy of the prediction

    Args:
        out (torch.tensor): model output
        truth (torch.tensor): ground truth output

    Returns:
        (float): accuracy
    """
    return np.sum(out == truth) / float(truth.size)
