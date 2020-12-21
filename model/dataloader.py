import os
import pdb
import pandas as pd
import pickle as pkl
import torch
from torch.utils.data import Dataset, DataLoader


class DetectionDataset(Dataset):
    def __init__(self, dslice):
        """Load the datasets
        
        Args:
            dslice: data slice (train, test, or val)
        """

        dirname = os.path.join('/scratch/users/georgech/data/preprocessed_binary/',dslice)
        self.true = torch.load(os.path.join(dirname,'true.pth'))
        self.cond = torch.load(os.path.join(dirname,'cond.pth'))
        self.out = torch.load(os.path.join(dirname,'out.pth'))#.view([-1,1])

    def __len__(self):
        # Return the size of the dataset
        return self.out.size()[0]

    def __getitem__(self, idx):
        """Access data sample given an index

        Args:
            idx: index of data sample
        """

        return self.cond[idx], self.true[idx], self.out[idx]


