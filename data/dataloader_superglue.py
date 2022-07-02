from transformers import DataCollatorForLanguageModeling
from functools import partial
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
import data.utils_datasets as utils
import torch
import ipdb

class DataSuperglue():

  def __init__(self, _lm, _data_name, _dataset_class, _device):
    self.data_collator = DataCollatorForLanguageModeling(
          tokenizer=_lm.tokenizer, mlm=True, mlm_probability=0.15)
    self.lm = _lm  
    self.device = _device  
    self.data_name = _data_name  
    self.dataset_class = _dataset_class
    self.data = utils.get_data(_data_name)
    self.BATCH_SIZE = 8

  def preprocess_batch(self, _batch):      

      negative, positive, label = list(zip(*_batch))

      positive_tokenized = self.lm.tokenizer(list(positive), padding=True, truncation=True, return_tensors="pt")
      negative_tokenized = self.lm.tokenizer(list(negative), padding=True, truncation=True, return_tensors="pt")

      batch = (positive_tokenized, negative_tokenized)

      if self.lm.is_mlm:
        positive_input, positive_label = self.data_collator(tuple(positive_tokenized)).values()
        negative_input,negative_label = self.data_collator(tuple(negative_tokenized)).values()
        batch += (positive_input, negative_input)
      
      labels = torch.Tensor(label)
      batch += labels,

      return batch

  def get_db_dataloaders(self):
      return (self.get_dataloader(self.data["train"].data),
      self.get_dataloader(self.data["validation"].data),
      self.get_dataloader(self.data["test"].data))


  def get_dataloader(self, _data):
      # ipdb.set_trace()
      ds = self.dataset_class(_data[:20], self.lm)
      return DataLoader(ds, batch_size=self.BATCH_SIZE, shuffle=True, collate_fn=partial(ds.preprocess_batch, _data_collator=self.data_collator))