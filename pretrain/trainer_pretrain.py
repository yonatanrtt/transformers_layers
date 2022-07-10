from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from pretrain.model_pretrain import PreTrainModel
from shared.lm_class import LM
import shared.utils as utils
from data.dataloader_superglue import DataSuperglue
import shared.constants as constants
import ipdb

class PretrainTrainer():

  def __init__(self, _task):
    _db_name, _db_class, _model = utils.init_task(_task)
    self.lm: LM = LM(_is_pretrain=True)
    self.db_name = _db_name
    self.train_dataloader, self.val_dataloader, self.test_dataloader = DataSuperglue(self.lm, _db_name, _db_class).get_db_dataloaders()
    self.model = PreTrainModel(self.lm, _task)
    self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
    self.N_EPOCH = constants.N_EPOCH
    self.writer = SummaryWriter()
    self.N_LOG_STEPS = 100


  def train(self):
    self.model.train()
    for epoch in range(self.N_EPOCH):
      for step, batch in enumerate(self.train_dataloader):
          self.optimizer.zero_grad()
          loss, losses = self.model(batch)
          (output_mlm_loss, layers_loss_sum, layers_distance_sum, fake_loss, real_loss) = losses
          self.writer.add_scalar(f"{self.db_name}/Loss/train", loss, epoch, step)
          if step % self.N_LOG_STEPS == 0:
              print(f"loss: {loss}")
          loss.backward()
          self.optimizer.step()
    print(constants.PRINT_COLOR, f"mlm.: {output_mlm_loss:.2f}, layers classifier: {layers_loss_sum:.2f}, layers_distance: {layers_distance_sum:.2f}, gan real: {real_loss:.2f}, gan fake: {fake_loss:.2f}")
    self.model.save_lm()