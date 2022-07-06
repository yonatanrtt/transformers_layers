from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from pretrain.model_pretrain import PreTrainModel
from shared.lm_class import LM
import shared.utils as utils
from data.dataloader_superglue import DataSuperglue
import shared.constants as constants

class PretrainTrainer():

  def __init__(self, _task):
    _db_name, _db_class, _model = utils.init_task(_task)
    self.lm: LM = LM(_is_pretrain=True)
    self.db_name = _db_name
    self.train_dataloader, self.val_dataloader, self.test_dataloader = DataSuperglue(self.lm, _db_name, _db_class).get_db_dataloaders()
    self.model = PreTrainModel(self.lm)
    self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
    self.N_EPOCH = constants.N_EPOCH
    self.writer = SummaryWriter()


  def train(self):
    self.model.train()
    for epoch in range(self.N_EPOCH):
      for step, batch in enumerate(self.train_dataloader):
          self.optimizer.zero_grad()
          loss = self.model(batch)
          self.writer.add_scalar(f"{self.db_name}/Loss/train", loss, epoch, step)
          loss.backward()
          self.optimizer.step()
    self.model.save_lm()