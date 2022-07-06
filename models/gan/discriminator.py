import torch.nn as nn
import torch
import ipdb

class Discriminator():

    def __init__(self, _lm):
        # ipdb.set_trace()
        hidden_size = _lm.config.hidden_size
        num_hidden_layers = _lm.config.num_hidden_layers + 1 # + embedding layer
        dropout_rate = _lm.config.hidden_dropout_prob
        self.GENERATED_LAYER = num_hidden_layers + 1 # + not real - generated
        self.layers = nn.Sequential(
          nn.Linear(_lm.config.hidden_size, hidden_size),
          nn.LeakyReLU(),
          nn.Dropout(dropout_rate),
          nn.Linear(hidden_size, self.GENERATED_LAYER)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        print(f"self.GENERATED_LAYER: {self.GENERATED_LAYER}")

    def forward(self, _input, _label, _is_generated):

        # ipdb.set_trace()
        if _is_generated:
          _label = self.GENERATED_LAYER - 1
        output = self.layers(_input)
        b_output = output.view(1, output.size(0))
        loss = self.loss_fn(b_output, torch.Tensor([_label]).long())
        return loss
        # if _label == self.GENERATED_LAYER or output == self.GENERATED_LAYER:
        #  return loss
        # else:
        #   return None




