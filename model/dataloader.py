# -*- coding: utf-8 -*-
"""
dataloader.py

A class for loading a sample of the sky conditions,
true galaxy magnitudes, and whether the galaxies were detected.

Author: George Halal
Email: halalgeorge@gmail.com
"""


__author__ = "George Halal"
__email__ = "halalgeorge@gmail.com"


import os
import pdb

import pandas as pd
import pickle as pkl
import torch
from torch.utils.data import Dataset, DataLoader


class DetectionDataset(Dataset):
    def __init__(self, dslice: str) -> None:
        """Load the datasets
        
        Args:
            dslice (str): data slice (one of train, test, or val)
        """
        dirname = os.path.join(
            "/scratch/users/georgech/data/preprocessed_binary/", dslice)
        self.true = torch.load(os.path.join(dirname, "true.pth"))
        self.cond = torch.load(os.path.join(dirname, "cond.pth"))
        self.out = torch.load(os.path.join(dirname, "out.pth"))

    def __len__(self) -> int:
        """Return the size of the dataset"""
        return self.out.size()[0]

    def __getitem__(self, idx: int) -> tuple[
            torch.tensor, torch.tensor, torch.tensor]:
        """Access data sample given an index.

        Args:
            idx (int): index of data sample

        Returns:
            (tuple[torch.tensor, torch.tensor, torch.tensor]):
                sky conditions, true magnitudes, and whether the
                galaxies were detected.
        """
        return self.cond[idx], self.true[idx], self.out[idx]


