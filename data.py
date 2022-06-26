from datasets import load_dataset, concatenate_datasets
import wandb
import pandas as pd
import torch
from torch.utils.data import DataLoader
from functools import partial
from transformers import DataCollatorForLanguageModeling


def get_datasets(): 
    dataset = "super_glue"
    copa = load_dataset(dataset, "copa")
    cb = load_dataset(dataset, "cb")
    rte = load_dataset(dataset, "rte")
    wic = load_dataset(dataset, "wic")
    return copa, cb, rte, wic


class CopaDataset(torch.utils.data.Dataset):
    def __init__(self, _data):
        self.premise = list(_data["premise"])
        self.choice1 = list(_data["choice1"]) 
        self.choice2 = list(_data["choice2"])  
        self.question = list(_data["question"])   
        self.label = list(_data["label"])

    def __getitem__(self, _idx):
        connector = "because" if self.question[_idx] == "cause" else "so"
        answers = [self.choice1[_idx], self.choice2[_idx]]

        positive_answers_idx = self.label[_idx]
        negative_answers_idx = 1 - positive_answers_idx

        positive = connector.join((self.premise[_idx], answers[positive_answers_idx]))
        negative = connector.join((self.premise[_idx], answers[negative_answers_idx]))

        return negative, positive

    def __len__(self):
        return len(self.label)


def preprocess_batch(_batch, _lm, _device):
   
    data_collator = DataCollatorForLanguageModeling(
          tokenizer=_lm.tokenizer, mlm=True, mlm_probability=0.15)  

    negative, positive = list(zip(*_batch))
    positive_tokenized = _lm.tokenizer.encode(positive, padding=True, truncation=True, return_tensors="pt")
    negative_tokenized = _lm.tokenizer.encode(negative, padding=True, truncation=True, return_tensors="pt")

    positive_input, positive_label = data_collator(tuple(positive_tokenized)).values()
    negative_input,negative_label = data_collator(tuple(negative_tokenized)).values()

    return positive_tokenized.to(_device), negative_tokenized.to(_device), positive_input.to(_device), negative_input.to(_device)


def get_dataloader(_data, _lm, _device):
  ds = CopaDataset(_data)
  return DataLoader(ds, batch_size=2, shuffle=True, collate_fn=partial(preprocess_batch, _lm=_lm, _device=_device))
