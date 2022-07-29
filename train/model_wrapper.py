import torch.nn as nn 
import torch
import shared.utils as utils 
from train.gan.generator import Generator 
from train.gan.discriminator import Discriminator 
import random 
import shared.constants as constants

class ModelWrapper(nn.Module):

    def __init__(self, _lm, _task, _task_model, _seq_embed_type, _layer_mean_axis, _matrix_dist_func, _w_mlm=1,
        _w_layers=1, _w_distance=1, _w_fake=1, _W_real=1): 
        super(ModelWrapper, self).__init__() 
        self.lm = _lm 
        self.model = _lm.model_mlm
        self.layers_linear = nn.Linear(_lm.config.hidden_size, _lm.config. num_hidden_layers + 1)
        self.act = nn.Sigmoid() 
        self.dropout = nn.Dropout(p=self.lm.config.hidden_dropout_prob) 
        self.loss_fn = nn.CrossEntropyLoss() 
        self.device = utils.get_device() 
        self.generator = Generator(self.lm) 
        self.generator.to(self.device) 
        self.discriminator = Discriminator(self.lm) 
        self.discriminator.to(self.device) 
        self.task_model = _task_model(self.lm) 
        self.task = _task
        self.EPS = 1e-08
        self.W_MLM = _w_mlm 
        self.W_LAYERS =_w_layers 
        self.W_DISTANCE = _w_distance 
        self.W_FAKE = _w_fake 
        self.W_REAL = _W_real 
        self.is_freeze = False 
        self.seq_embed_type = _seq_embed_type 
        self.layer_mean_axis = _layer_mean_axis 
        self.matrix_dist_func = _matrix_dist_func
    
    def forward(self, _batch): 
        if len(_batch) == 4:
            _input_tokenized, _input_masked, _masked_label, _labels = _batch 
        else:
            _input_tokenized, _input_2_tokenized, _input_masked, _masked_label, _labels = _batch
        if self.model.training:
            return self.train_forward(_batch, _input_tokenized, _input_masked, _masked_label)
        else:
            output, supervised_loss, _labels = self.task_model(_batch) 
            return output, supervised_loss, _labels
    
    def train_forward(self, _batch, _input_tokenized, _input_masked, _masked_label):
        output_lm = self.model(_input_tokenized.input_ids, attention_mask=_input_tokenized.attention_mask)
        layers_classifiers, layers_distance, layers_real_gan = [], [], [] 
        # rand_idx = random.randrange(0, len(output_lm.hidden_states)) 
        for layer_idx, layer in enumerate(output_lm.hidden_states):
            seq_embed = utils.get_seq_embed(_input=_input_tokenized, _output=layer, _type=self.seq_embed_type)
            b_idx = random.randrange(0, seq_embed.size(0))
            layers_real_gan.append((seq_embed[b_idx], layer_idx + 1)) # 0 is for fake layer

            output = self.dropout(seq_embed) 
            output = self.layers_linear(output)

            layer_labels = torch.Tensor([layer_idx] * layer.size(0)).long().to(self.device)
            layer_loss = self.loss_fn(output, layer_labels)
            layers_classifiers.append(layer_loss)

            if layer_idx > 0 and self.matrix_dist_func != constants.MATRIX_DIST_ORTHOGONAL:
                if self.matrix_dist_func == constants.DIALOG_VECTORS:
                    prev_layer_seq_embed = utils.get_seq_embed(_input=_input_tokenized, _output=output_lm.hidden_states[layer_idx - 1],
                    _type=self.seq_embed_type)
                    dist_matrix = seq_embed @ prev_layer_seq_embed.T 
                    seq_embed_dist = torch.diagonal(dist_matrix).abs() 
                    dist = torch.mean(seq_embed_dist)
                    layers_distance.append(dist) 
                elif self.matrix_dist_func == constants.MATRIX_DIST_MUL:
                    sizes = layer.shape
                    prev_layer = output_lm. hidden_states[layer_idx - 1].view(sizes [0], sizes[-1], sizes[1])
                    dist = utils.matrix_dist_mul(layer, prev_layer, self.layer_mean_axis)
                    layers_distance. append(dist) 
                elif self.matrix_dist_func == constants.MATRIX_DIST_SUM:
                    prev_layer = output_lm.hidden_states[layer_idx - 1] 
                    dist = utils.matrix_dist_sum(layer, prev_layer)
                    layers_distance. append(dist) 
                elif self.matrix_dist_func == constants .MATRIX_DIST:
                    prev_layer = output_lm.hidden_states[layer_idx - 1] 
                    dist = utils.matrix_dist(layer, prev_layer)
                    layers_distance.append(dist) 
            
            if self.matrix_dist_func == constants. MATRIX_DIST_ORTHOGONAL:
                layers_distance.append(seq_embed)
        
        layers_loss_sum = sum(layers_classifiers) 
        if self.matrix_dist_func == constants.MATRIX_DIST_ORTHOGONAL:
            layers_distance_sum = utils.orthogonal_penalty(torch.stack(layers_distance))
        else:
            layers_distance_sum = sum(layers_distance)

        real_output, real_logit, real_prob, real_labels = self.discriminator(layers_real_gan)
        g_input = self.generator()
        fake_output, fake_logit, fake_prob, fake_labels = self.discriminator(g_input)
        
        d_real_loss = -1 * torch.mean(torch.log(1 - real_prob[:, constants. FAKE_LABEL] + self.EPS), axis=0)
        d_fake_loss = -1 * torch.mean(torch.log(fake_prob[:, constants. FAKE_LABEL] + self.EPS), axis=0)
        d_loss = d_real_loss + d_fake_loss
        
        g_fake_loss = -1 * torch.mean(torch.log(1 - fake_prob[:, constants. FAKE_LABEL] + self.EPS), axis=0)
        g_feat_match = torch.mean(torch.square(torch.mean(real_output, axis=0) - torch.mean(fake_output, axis=0)))
        g_loss = g_fake_loss + g_feat_match
        
        output_mlm = self.model(_input_masked, labels=_masked_label) 
        output, supervised_loss, _labels = self.task_model(_batch)
        
        if self.is_freeze:
            loss = (self.W_FAKE * g_loss) + (self.W_REAL * d_loss)
            losses = (supervised_loss, output_mlm.loss, layers_loss_sum, layers_distance_sum, g_loss, d_loss)
            output, _labels = [], [] 
        else:
            loss = supervised_loss + (self.W_MLM * output_mlm.loss) + (self.W_LAYERS * layers_loss_sum) + (
                self.W_DISTANCE * layers_distance_sum) + (self.W_FAKE * g_loss) + (self.W_REAL * d_loss)
            losses = (supervised_loss, output_mlm.loss, layers_loss_sum, layers_distance_sum, g_loss, d_loss)
        return loss, losses, output, _labels
    
    def save_lm(self):
        save_lm_path = constants.SAVE_LM_PATH + self.task 
        self.lm.tokenizer.save_pretrained(save_lm_path) 
        self.model.save_pretrained(save_lm_path)

    def set_is_freeze(self, _is_freeze):
        self.is_freeze = _is_freeze
