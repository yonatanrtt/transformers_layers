import torch.nn as nn
import torch
import ipdb

class Generator():

    def __init__(self, _lm):

        self.hidden_size = _lm.config.hidden_size
        self.dropout_rate = _lm.config.hidden_dropout_prob
        self.layers = nn.Sequential(
          nn.Linear(self.hidden_size, self.hidden_size),
          nn.LeakyReLU(),
          nn.Dropout(self.dropout_rate),
          nn.Linear(self.hidden_size, self.hidden_size)
        )

    def forward(self):

        # ipdb.set_trace()
        inputs = torch.rand(self.hidden_size)
        output = self.layers(inputs)
        return output


