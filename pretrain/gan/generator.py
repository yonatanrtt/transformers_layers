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
        )
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self):

        # ipdb.set_trace()
        inputs = torch.rand(self.hidden_size)   
        # GAN-BERT - uses loop here for self.layers(inputs)               
        output = self.layers(inputs)
        # seperate from the loop
        output = self.linear(output)
        return output


