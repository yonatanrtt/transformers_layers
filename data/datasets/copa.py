import pandas as pd
import torch
from torch.utils.data import DataLoader
from functools import partial

class CopaDataset(torch.utils.data.Dataset):

    def __init__(self, _data, _lm):
        self.premise = list(_data["premise"])
        self.choice1 = list(_data["choice1"]) 
        self.choice2 = list(_data["choice2"])  
        self.question = list(_data["question"])   
        self.label = list(_data["label"])
        self.lm = _lm

    def __getitem__(self, _idx):
        # connector = "because" if self.question[_idx] == "cause" else "so"
        connector = f" {self.lm.tokenizer.sep_token} "
        answers = [self.choice1[_idx], self.choice2[_idx]]

        positive_answers_idx = self.label[_idx]
        negative_answers_idx = 1 - positive_answers_idx
        
        first_attr = [self.premise[_idx]] * 2
        second_attr = [answers[positive_answers_idx], answers[negative_answers_idx]]

        if self.question[_idx] == "cause":
              first_attr, second_attr = second_attr, first_attr

        positive = connector.join(first_attr[0], second_attr[0])
        negative = connector.join(first_attr[1], second_attr[1])

        label = self.label[_idx]

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

