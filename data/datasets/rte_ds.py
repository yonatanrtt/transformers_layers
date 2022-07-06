import pandas as pd
import torch
from torch.utils.data import DataLoader
from functools import partial
import ipdb

class RteDataset(torch.utils.data.Dataset):

    def __init__(self, _data, _lm):
        self.premise = list(_data["premise"])
        self.hypothesis = list(_data["hypothesis"]) 
        self.label = list(_data["label"])
        self.lm = _lm

    def __getitem__(self, _idx):

        premise = self.premise[_idx].as_py()
        hypothesis = self.hypothesis[_idx].as_py()

        inputs = premise+self.lm.tokenizer.sep_token+hypothesis
        label = self.label[_idx].as_py()

        return inputs, label

    def __len__(self):
        return len(self.label)


    def preprocess_batch(self, _batch, _data_collator):      

      inputs, label = list(zip(*_batch))

      inputs_tokenized = self.lm.tokenizer(list(inputs), padding=True, truncation=True, return_tensors="pt")
      batch = inputs_tokenized,

      if self.lm.is_mlm:
        inputs_input, inputs_label =  _data_collator(tuple(inputs_tokenized.input_ids)).values()
        batch += inputs_tokenized,
      
      labels = torch.Tensor(label)
      batch += labels,

      return batch
