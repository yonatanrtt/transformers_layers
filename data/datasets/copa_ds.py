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
        
        first_attr = [self.premise[_idx].as_py()] * 2
        second_attr = answers

        if self.question[_idx].as_py() == "cause":
              first_attr, second_attr = second_attr, first_attr

        input_1 = first_attr[0] + connector + second_attr[0]
        input_2 = first_attr[1] + connector + second_attr[1]

        positive = input_1 if label == 0 else input_2

        return input_1, input_2, positive, label

    def __len__(self):
        return len(self.label)

    def preprocess_batch(self, _batch, _data_collator):  

      input_1, input_2, positive, label = list(zip(*_batch))

      input_1_tokenized = self.lm.tokenizer(list(input_1), padding=True, truncation=True, return_tensors="pt")
      input_2_tokenized = self.lm.tokenizer(list(input_2), padding=True, truncation=True, return_tensors="pt")

      batch = (input_1_tokenized, input_2_tokenized)

      if self.lm.is_mlm:
        positive_tokenized = self.lm.tokenizer(list(positive), padding=True, truncation=True, return_tensors="pt")
        masked_input =  _data_collator(tuple(positive_tokenized.input_ids)).values()
        batch += masked_input,
      
      labels = torch.Tensor(label)
      batch += labels,

      return batch

