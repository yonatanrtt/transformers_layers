from datasets import load_dataset, concatenate_datasets
import wandb
import pandas as pd
import torch
from torch.utils.data import DataLoader

def get_datasets(): 
    dataset = "super_glue"
    copa = load_dataset(dataset, "copa")
    cb = load_dataset(dataset, "cb")
    rte = load_dataset(dataset, "rte")
    wic = load_dataset(dataset, "wic")
    return copa, cb, rte, wic


class CopaDataset(torch.utils.data.Dataset):
    def __init__(self, _data, _tokenizer):
        self.premise = list(_data["premise"])
        self.choice1 = list(_data["choice1"]) 
        self.choice2 = list(_data["choice2"])  
        self.question = list(_data["question"])   
        self.label = list(_data["label"])  
        self.tokenizer = _tokenizer
        self.tokenizer = _tokenizer

    def __getitem__(self, _idx):
        connector = "because" if self.question[_idx] == "cause" else "so"
        answers = [self.choice1[_idx], self.choice2[_idx]]

        positive_answers_idx = self.label[_idx]
        negative_answers_idx = 1 - positive_answers_idx

        positive = connector.join((self.premise[_idx], answers[positive_answers_idx]))
        negative = connector.join((self.premise[_idx], answers[negative_answers_idx]))

        positive_tokenized = self.tokenizer.encode(positive, padding="max_length", truncation=True, return_tensors="pt")[0]
        negative_tokenized = self.tokenizer.encode(negative, padding="max_length", truncation=True, return_tensors="pt")[0]


        return positive_tokenized, negative_tokenized

    def __len__(self):
        return len(self.label)


def get_dataloader(_data, _tokenizer):
  ds = CopaDataset(_data, _tokenizer)
  return DataLoader(ds, batch_size=5, shuffle=True)
