import pandas as pd
import torch
from torch.utils.data import DataLoader
from functools import partial

class RteDataset(torch.utils.data.Dataset):

    def __init__(self, _data, _lm):
        self.premise = list(_data["premise"])
        self.hypothesis = list(_data["hypothesis"]) 
        self.label = list(_data["label"])
        self.lm = _lm

    def __getitem__(self, _idx):
        
        premise = self.premise[_idx]
        hypothesis = self.hypothesis[_idx]
        label = self.label[_idx]

        return premise, hypothesis, label

    def __len__(self):
        return len(self.label)
