from transformers import DataCollatorForLanguageModeling
from functools import partial
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
import datasets.utils_datasets as utils
import torch
def superglue_dataloader():

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

      positive_tokenized = self.lm.tokenizer.encode(positive, padding=True, truncation=True, return_tensors="pt")
      negative_tokenized = self.lm.tokenizer.encode(negative, padding=True, truncation=True, return_tensors="pt")

      positive_input, positive_label = self.data_collator(tuple(positive_tokenized)).values()
      negative_input,negative_label = self.data_collator(tuple(negative_tokenized)).values()

      labels = torch.Tensor(label)

      return labels, positive_tokenized.to(self.device), negative_tokenized.to(self.device), positive_input.to(self.device), negative_input.to(self.device)

  def get_dataloaders(self):
      return (self.get_dataloader(self.data["train"].data),
      self.get_dataloader(self.data["val"].data),
      self.get_dataloader(self.data["test"].data))


  def get_dataloader(self, _data):
      ds = self.dataset_class(_data, self.lm)
      return DataLoader(ds, batch_size=self.BATCH_SIZE, shuffle=True, collate_fn=partial(preprocess_batch))