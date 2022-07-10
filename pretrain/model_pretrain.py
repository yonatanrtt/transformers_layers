import torch
import torch.nn as nn
import ipdb
import random

from pretrain.gan.generator import Generator
from pretrain.gan.discriminator import Discriminator
import shared.constants as constants

class PreTrainModel(nn.Module):

    def __init__(self, _lm, _task):
        super(PreTrainModel, self).__init__()
        self.lm = _lm
        self.model = _lm.model_mlm
        self.layers_linear = nn.Linear(_lm.config.hidden_size, _lm.config.num_hidden_layers+1)
        self.act = nn.Sigmoid()
        self.dropout =  nn.Dropout(p=self.lm.config.hidden_dropout_prob)
        self.loss_fn = nn.CrossEntropyLoss()
        self.generator = Generator(self.lm)
        self.discriminator = Discriminator(self.lm)
        self.task = _task
        self.EPS = 1e-08
        self.W_MLM = 1
        self.W_LAYERS = 1
        self.W_DISTANCE = 1
        self.W_FAKE = 1
        self.W_REAL= 1

    def forward(self, _batch):
        if len(_batch) == 3:
          _input_tokenized, _input_masked, _labels = _batch
        else:
          _input_tokenized, _input_2_tokenized, _input_masked, _labels = _batch
        output_lm = self.model(_input_tokenized.input_ids, attention_mask=_input_tokenized.attention_mask)

        layers_classifaiers, layers_distance = [], []
        rand_idx = random.randrange(0, len(output_lm.hidden_states))

        
        for layer_idx, layer in enumerate(output_lm.hidden_states):
          cls = layer[:, 0, :]
          if layer_idx == rand_idx:
            b_idx = random.randrange(0, cls.size(0))
            real_output, real_logit, real_prob = self.discriminator.forward(cls[b_idx], layer_idx + 1)
          output = self.dropout(cls)
          output = self.layers_linear(output)
          layer_labels = torch.Tensor([layer_idx]*layer.size(0)).long()
          layer_loss =  self.loss_fn(output, layer_labels)    
          layers_classifaiers.append(layer_loss)
          if layer_idx > 0:
            prev_layer_cls = output_lm.hidden_states[layer_idx - 1][:, 0, :]
            dist_matrix = cls @ prev_layer_cls.T
            cls_dist = torch.diagonal(dist_matrix)
            dist_mean = torch.mean(cls_dist)
            layers_distance.append(dist_mean)
        layers_loss_sum = sum(layers_classifaiers)
        layers_distance_sum = sum(layers_distance)
      
        g_input = self.generator.forward()
        fake_output, fake_logit, fake_prob = self.discriminator.forward(g_input, constants.FAKE_LABEL)

        real_loss = -1 * torch.log(1 - real_prob[:, constants.FAKE_LABEL] + self.EPS)
        real_loss = real_loss[0]
        fake_loss = -1 * torch.log(fake_prob[:, constants.FAKE_LABEL] + self.EPS)
        fake_loss = fake_loss[0]
        output_mlm = self.model(_input_masked, labels=_input_tokenized.input_ids)
        
        loss = ( self.W_MLM * output_mlm.loss ) + ( self.W_LAYERS * layers_loss_sum ) + ( self.W_DISTANCE * layers_distance_sum ) + ( self.W_FAKE * fake_loss ) + ( self.W_REAL * real_loss )
        losses = (output_mlm.loss, layers_loss_sum, layers_distance_sum, fake_loss, real_loss)
        return loss, losses

    def save_lm(self):
      save_lm_path = constants.SAVE_LM_PATH + self.task
      self.lm.tokenizer.save_pretrained(save_lm_path)
      self.model.save_pretrained(save_lm_path)

