from transformers import get_linear_schedule_with_warmup 
from torch.optim import AdamW
from torch.utils. tensorboard import SummaryWriter 
import torch
from shared.lm_class import LM 
import shared.utils as utils 
import shared.constants as constants 
from data.dataloader_superglue import DataSuperglue 
from train.model_wrapper import ModelWrapper

class SuperGlueTrainer:
    
    def __init__(self, _task, _lm_path, _data_limit, _is_clip_grad_norm, _seq_embed_type, _layer_mean_axis,
        _matrix_dist_func, _w_mlm=0, _w_layers=0, _w_distance=0, _w_fake=0, _w_real=0): 
        _db_name, _db_class, _model = utils.init_task(_task) 
        self.lm_path = _lm_path 
        self.lm: LM = LM(_lm_path=_lm_path, _is_pretrain=True) 
        self.task = _db_name
        self.train_dataloader, self.val_dataloader, self.test_dataloader = DataSuperglue(self.lm, _db_name, _db_class, _data_limit).get_db_dataloader()
        self.model = ModelWrapper(_lm=self.lm, _task=_task, _task_model=_model, _seq_embed_type=_seq_embed_type,
        _layer_mean_axis=_layer_mean_axis, _matrix_dist_func=_matrix_dist_func, _w_mlm=_w_mlm,
        _w_layers=_w_layers, _w_distance=_w_distance, _w_fake=_w_fake, _W_real=_w_real)
        self.model.to(utils.get_device())
        self. BASE_LM_ARCH = [self.lm.model_mlm.roberta.embeddings, *self.lm.model_mlm.roberta.encoder.layer[:]]
        self.optimizer = AdamW(self.model.parameters(), lr=10e-5) 
        self.hparams = dict(
                _data_limit=_data_limit, 
                _is_clip_grad_norm=_is_clip_grad_norm, 
                _seq_embed_type=_seq_embed_type, 
                _layer_mean_axis=_layer_mean_axis, 
                _matrix_dist_func=_matrix_dist_func, 
                _w_mlm=_w_mlm, 
                _w_layers=_w_layers, 
                _w_distance=_w_distance, 
                _w_fake=_w_fake, 
                _w_real=_w_real
        )
        log_name = "_".join([str(i) for i in list(self.hparams.values())])
        self.writer = SummaryWriter(f"{constants .SAVE_RESULTS}/Y_train/{self.task}/{log_name}/run")
        self.FREEZE_EPOCHS = 5 
        self.N_EPOCH = constants.N_EPOCH + self. FREEZE_EPOCHS 
        self.total_steps = len(self.train_dataloader) * (constants.N_EPOCH * 2)
        self. FREEZE_LAYERS = [f"model.encoder.layer.{layer_idx + 1}." for layer_idx in
        range(self.lm.config.num_hidden_layers)] 
        self.scheduler = None 
        self.is_clip_grad_norm = _is_clip_grad_norm 
        self.LOG_STEPS = 50 
    
    def train(self):
        self.model.train() 
        print("*" * 130 + "\n") 
        print(f"lm path: {self.lm_path}") 
        for epoch in range(self.N_EPOCH):
            is_freeze = epoch < self.FREEZE_EPOCHS 
            for name, param in self.model.named_parameters():
                if any(name.startswith(layer_str) for layer_str in self.FREEZE_LAYERS):
                    param.requires_grad = is_freeze 
            if is_freeze is False and self.scheduler is None:
                self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0,
            num_training_steps=self.total_steps)
            self.model.set_is_freeze(is_freeze) 
            for step, batch in enumerate(self.train_dataloader):
                self.model.train() 
                self.optimizer.zero_grad() 
                loss, losses, output, labels = self.model(batch) 
                loss.backward() 
                if self.is_clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) 
                if step % self. LOG_STEPS:
                    self.evaluation(losses, loss, epoch, step) 
                self.optimizer.step() 
                if is_freeze is False:
                    self.scheduler.step() 
            eval_accuracy = self.evaluation(losses, loss, epoch, step + 1)
            print(constants.PRINT_COLOR, f"{self.task} - accuracy: {eval_accuracy:.2f}%", constants.PRINT_END_COLOR)  
        self.set_hparams(losses, loss, eval_accuracy)
        print("*" * 130 + "\n") 
        return eval_accuracy


    def evaluation(self, _losses, _loss, _epoch=0, _step=0):
        log_step = _epoch * len(self.train_dataloader) + _step 
        self.print_losses(_losses, _loss, log_step) 
        self.model.eval() 
        accuracy = 0 
        for step, batch in enumerate(self.val_dataloader): 
            with torch.no_grad():
                output, supervised_loss, labels = self.model(batch) 
                batch_success = output.argmax(1).eq(labels).sum().item()
                accuracy += batch_success / len(labels) 
        accuracy = (accuracy / len(self.val_dataloader)) * 100 
        self.writer.add_scalar(f"Val/accuracy", accuracy, log_step) 
        return accuracy
    
    def print_losses(self, _losses, _loss, _log_step):
        (supervised_loss, output_mlm_loss, layers_loss_sum, layers_distance_sum, g_loss, d_loss) = _losses
        self.writer.add_scalar(f"Train/loss", _loss, _log_step)
        self.writer.add_scalar(f"Train/ supervised_loss", supervised_loss, _log_step)
        self.writer.add_scalar(f"Train/output_mlm_loss", output_mlm_loss, _log_step)
        self.writer.add_scalar(f"Train/layers_loss_sum", layers_loss_sum, _log_step)
        self.writer.add_scalar(f"Train/layers_distance_sum", layers_distance_sum, _log_step)
        self.writer.add_scalar(f"Train/g_loss", g_loss, _log_step)
        self.writer.add_scalar(f"Train/d_loss", d_loss, _log_step) 
    
    def set_hparams(self, _losses, _train_loss, _eval_accuracy):
        (supervised_loss, output_mlm_loss, layers_loss_sum, layers_distance_sum, g_loss, d_loss) = _losses
        self.writer.add_hparams(self.hparams, metric_dict={
          "Val/accuracy": _eval_accuracy, 
          "Train/loss": _train_loss, 
          "Train/supervised_loss": supervised_loss, 
          "Train/output_mlm_loss": output_mlm_loss, 
          "Train/layers_loss_sum": layers_loss_sum, 
          "Train/layers_distance_sum": layers_distance_sum, 
          "Train/g_loss": g_loss, 
          "Train/d_loss": d_loss
        })

    def test(self):
        self.model.eval() 
        accuracy = 0 
        for step, batch in enumerate(self.test_dataloader): 
            with torch.no_grad():
                output, loss, labels = self.model(batch)
                batch_success = torch. Tensor(output).argmax(1).eq(torch. Tensor(labels)). sum().item()
                batch_accuracy = batch_success / len(batch[0].input_ids)
                accuracy += batch_accuracy 
        print(f"{self.task} - test result: {accuracy / len(self.val_dataloader): .2f}%")
