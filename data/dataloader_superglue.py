from transformers import DataCollatorForLanguageModeling
from functools import partial
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
import torch
import ipdb

import shared.utils as utils
import shared.constants as constants
import data.utils_datasets as utils_datasets

class DataSuperglue():

  def __init__(self, _lm, _data_name, _dataset_class, _data_limit = None):
    self.data_collator = DataCollatorForLanguageModeling(
          tokenizer=_lm.tokenizer, mlm=True, mlm_probability=0.15)
    self.lm = _lm  
    self.data_limit = _data_limit
    self.device = utils.get_device()
    self.data_name = _data_name  
    self.dataset_class = _dataset_class
    self.data = utils_datasets.get_data(_data_name)
    self.BATCH_SIZE = 8

  def get_db_dataloaders(self):
      return (self.get_dataloader(self.data["train"]),
      self.get_dataloader(self.data["validation"]),
      self.get_dataloader(self.data["test"]))


  def get_dataloader(self, _data):
      ipdb.set_trace()
      data = _data
      if self.data_limit is not None:
        data = data.shuffle(seed=constants.SEED)
        data = data[:self.data_limit]
      data = data.data[:8]
      ds = self.dataset_class(data, self.lm)
      return DataLoader(ds, batch_size=self.BATCH_SIZE, shuffle=True, collate_fn=partial(ds.preprocess_batch, _data_collator=self.data_collator))