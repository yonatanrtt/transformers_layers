from transformers import get_linear_schedule_with_warmup 
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter 
import torch 
from torch.utils.tensorboard.summary import hparams
from shared.lm_class import LM 
import shared.utils as utils 
import shared.constants as constants 
from data.dataloader_superglue import DataSuperglue

class SuperGlueTrainerBaseline:
    
    def __init__(self, _task, _lm_path=None, _data_limit=None):
        _db_name, _db_class, _model = utils.init_task_baseline(_task) 
        self.lm_path = _lm_path 
        self.lm: LM = LM(_lm_path=_lm_path, _is_pretrain=False) 
        self.db_name = _db_name
        self.train_dataloader, self.val_dataloader, self.test_dataloader = DataSuperglue(self.lm, _db_name, _db_class,
        _data_limit).get_db_dataloader() 
        self.model = _model(self.lm) 
        self.model.to(utils.get_device()) 
        self.optimizer = AdamW(self.model.parameters(), lr=10e-5) 
        self.N_EPOCH = constants. N_EPOCH
        self.writer = SummaryWriter(f"{constants.SAVE_RESULTS}/baseline/{self.db_name}/run")
        total_steps = len(self.train_dataloader) * self.N_EPOCH
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
             num_warmup_steps=0, 
             num_training_steps=total_steps)
        self.LOG_STEPS = 50 
        self.hparams = dict(
            _data_limit=_data_limit, 
            _is_clip_grad_norm=-1, 
            _seq_embed_type=-1, 
            _layer_mean_axis=-1, 
            _matrix_dist_func=-1, 
            _w_mlm=-1, 
            _w_layers=-1, 
            _W_distance=-1,
            _W_fake=-1, 
            _W_real=-1
        )
    
    def train(self):
        self.model.train() 
        print("*" * 130 + "\n") 
        print(f"lm path: {self.lm_path}") 
        for epoch in range(self.N_EPOCH): 
            for step, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad() 
                output, loss, labels = self.model(batch) 
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) 
                if step % self.LOG_STEPS:
                    log_step = epoch * len(self.train_dataloader) + step 
                    self.writer.add_scalar("Train/loss", loss, log_step) 
                    self.evaluation(log_step) 
                self.optimizer.step()
                self.scheduler.step() 
            log_step = epoch * len(self.train_dataloader) + step + 1
            eval_accuracy = self.evaluation(log_step) 
            print(constants.PRINT_COLOR, f"{self.db_name} - accuracy: {eval_accuracy:.2f}%", constants.PRINT_END_COLOR) 
        print("*" * 130 + "\n") 
        self.writer.add_hparams(self.hparams, metric_dict={"Val/accuracy": eval_accuracy, "Train/loss": loss})
        return eval_accuracy
    
    def evaluation(self, _log_step):
        self.model.eval() 
        accuracy = 0 
        for step, batch in enumerate(self.val_dataloader): 
            with torch.no_grad():
                output, loss, labels = self.model(batch)
                batch_success = output.argmax(1).eq(labels).sum().item()
                accuracy += batch_success / len(labels) 
        accuracy = (accuracy / len(self.val_dataloader)) * 100
        self.writer.add_scalar(f"Val/accuracy", accuracy, _log_step) 
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
        print(f"{self.db_name} - test result: {accuracy / len(self.test_dataloader):.2f}%")
