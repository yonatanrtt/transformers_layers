import pandas as pd
import torch
from torch.utils.data import DataLoader
from functools import partial
import ipdb

class CopaDataset(torch.utils.data.Dataset):

    def __init__(self, _data, _lm):
        self.premise = list(_data["premise"])
        self.choice1 = list(_data["choice1"]) 
        self.choice2 = list(_data["choice2"])  
        self.question = list(_data["question"])   
        self.label = list(_data["label"])
        self.lm = _lm

    def __getitem__(self, _idx):

        connector = self.lm.tokenizer.sep_token
        answers = [self.choice1[_idx].as_py(), self.choice2[_idx].as_py()]

        label = self.label[_idx].as_py()

        positive_answers_idx = label
        negative_answers_idx = 1 - positive_answers_idx
        
        first_attr = [self.premise[_idx].as_py()] * 2
        second_attr = [answers[positive_answers_idx], answers[negative_answers_idx]]

        if self.question[_idx].as_py() == "cause":
              first_attr, second_attr = second_attr, first_attr

        positive = first_attr[0] + connector + second_attr[0]
        negative = first_attr[1] + connector + second_attr[1]

        return negative, positive, label

    def __len__(self):
        return len(self.label)

    def preprocess_batch(self, _batch, _data_collator):  

      negative, positive, label = list(zip(*_batch))

      positive_tokenized = self.lm.tokenizer(list(positive), padding=True, truncation=True, return_tensors="pt")
      negative_tokenized = self.lm.tokenizer(list(negative), padding=True, truncation=True, return_tensors="pt")

      batch = (positive_tokenized, negative_tokenized)

      if self.lm.is_mlm:
        positive_input, positive_label = _data_collator(tuple(positive_tokenized)).values()
        negative_input,negative_label = _data_collator(tuple(negative_tokenized)).values()
        batch += (positive_input, negative_input)
      
      labels = torch.Tensor(label)
      batch += labels,

      return batch

