from transformers import AdamW
from torch.utils.tensorboard import SummaryWriter
import torch

from shared.lm_class import LM
import shared.utils as utils
import shared.constants as constants
from data.dataloader_superglue import DataSuperglue

class SuperGlueTrainer():

  def __init__(self, _task, _lm_path=None, _data_limit=None):
    _db_name, _db_class, _model = utils.init_task(_task)
    self.lm_path = _lm_path
    self.lm: LM = LM(_lm_name=_lm_path, _is_pretrain=False)
    self.db_name = _db_name
    self.train_dataloader, self.val_dataloader, self.test_dataloader = DataSuperglue(self.lm, _db_name, _db_class, _data_limit).get_db_dataloaders()
    self.model = _model(self.lm)
    self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
    self.N_EPOCH = constants.N_EPOCH
    self.writer = SummaryWriter()


  def train(self):
    # ipdb.set_trace()
    self.model.train()
    print("*" * 130 + "\n")
    print(f"lm path: {self.lm_path}")
    eval_accuracy = self.evaluation(0, 0)
    for epoch in range(self.N_EPOCH):
      for step, batch in enumerate(self.train_dataloader):
          self.optimizer.zero_grad()
          output, loss, labels = self.model(batch)
          self.writer.add_scalar(f"{self.db_name}/Loss/train", loss, epoch, step)
          loss.backward()
          self.optimizer.step()
      eval_accuracy = self.evaluation(epoch, step + 1)
    print("*" * 130 + "\n")
    return eval_accuracy
  
  def evaluation(self, _epoch=0, _step=0):
    self.model.eval()
    accuracy = 0
    for step, batch in enumerate(self.val_dataloader):
        with torch.no_grad():
          output, loss, labels = self.model(batch)
          batch_success = torch.Tensor(output).argmax(1).eq(torch.Tensor(labels)).sum().item() 
          batch_accuracy = batch_success / len(batch[0].input_ids)
          accuracy += batch_accuracy
    print(constants.PRINT_COLOR, f"{self.db_name} - accuracy: {accuracy / len(self.val_dataloader):.2f}%")
    self.writer.add_scalar(f"{self.db_name}/accuracy/val", accuracy, _epoch, _step)
    return accuracy

  def test(self):
    self.model.eval()
    accuracy = 0
    for step, batch in enumerate(self.test_dataloader):
        with torch.no_grad():
          output, loss, labels = self.model(batch)
          batch_success = torch.Tensor(output).argmax(1).eq(torch.Tensor(labels)).sum().item() 
          batch_accuracy = batch_success / len(batch[0].input_ids)
          accuracy += batch_accuracy
    print(f"{self.db_name} - test result: {accuracy / len(self.val_dataloader):.2f}%")