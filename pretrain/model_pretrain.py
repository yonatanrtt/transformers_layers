import torch
import torch.nn as nn
import ipdb
import random

from models.gan.generator import Generator
from models.gan.discriminator import Discriminator

class PreTrainModel(nn.Module):

    def __init__(self, _lm):
        super(PreTrainModel, self).__init__()
        self.lm = _lm
        self.model = _lm.model_mlm
        self.layers_linear = nn.Linear(_lm.config.hidden_size, _lm.config.num_hidden_layers+1)
        self.act = nn.Sigmoid()
        self.dropout =  nn.Dropout(p=self.lm.config.hidden_dropout_prob)
        self.loss_fn = nn.CrossEntropyLoss()
        self.generator = Generator(self.lm)
        self.discriminator = Discriminator(self.lm)

    def forward(self, _batch):
        _input_tokenized, _input_masked, _labels = _batch
        output_lm = self.model(_input_tokenized.input_ids, attention_mask=_input_tokenized.attention_mask)

        layers_classifaiers = []
        rand_idx = random.randrange(0, len(output_lm.hidden_states))

        
        for layer_idx, layer in enumerate(output_lm.hidden_states):
          cls = layer[:, 0, :]
          if layer_idx == rand_idx:
            b_idx = random.randrange(0, cls.size(0))
            dis_loss = self.discriminator.forward(cls[b_idx], layer_idx, False)
          output = self.dropout(cls)
          output = self.layers_linear(output)
          layer_labels = torch.Tensor([layer_idx]*layer.size(0)).long()
          layer_loss =  self.loss_fn(output, layer_labels)    
          layers_classifaiers.append(layer_loss)
        layers_loss_sum = sum(layers_classifaiers)
      
        gen = self.generator.forward()
        gen_dis_loss = self.discriminator.forward(gen, _label=None, _is_generated=True)

        output_mlm = self.model(_input_masked.input_ids, attention_mask=_input_masked.attention_mask, labels=_input_tokenized.input_ids)
        
        loss = output_mlm.loss + layers_loss_sum + dis_loss + gen_dis_loss
        print(f"loss: {loss:.2f}, output_mlm.loss: {output_mlm.loss:.2f}, layers_loss_sum: {layers_loss_sum:.2f}, layers_classifaiers: {layers_classifaiers[0]:.2f}, dis_loss: {dis_loss:.2f}, gen_dis_loss: {gen_dis_loss:.2f}")
        return loss

    def save_lm(self):
      self.lm.tokenizer.save_pretrained("/content/LM")
      self.model.save_pretrained("/content/LM")

        