import shared.constants as constants
import torch
import random
import numpy as np

def init_task(_task):
  db_name = _task
  if _task == constants.CB:
      from data.datasets.cb_ds import CbDataset
      from models.cb_model import CbModel
      db_class= CbDataset
      model = CbModel
  elif _task == constants.COPA:
      from data.datasets.copa_ds import CopaDataset
      from models.copa_model import CopaModel
      db_class= CopaDataset
      model = CopaModel
  else:
      from data.datasets.rte_ds import RteDataset
      from models.rte_model import RteModel
      db_name = constants.RTE
      db_class= RteDataset
      model = RteModel
  return db_name, db_class, model

def get_device():
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  return device

def set_all_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)