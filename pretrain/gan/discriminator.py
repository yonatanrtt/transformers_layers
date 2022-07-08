import torch.nn as nn
import torch
import ipdb

class Discriminator():

    def __init__(self, _lm):
        # ipdb.set_trace()
        hidden_size = _lm.config.hidden_size
        num_hidden_layers = _lm.config.num_hidden_layers + 1 # + embedding layer
        dropout_rate = _lm.config.hidden_dropout_prob
        n_layers = num_hidden_layers + 1 # + 1 for not real - generated
        self.layers = nn.Sequential(
          nn.Linear(_lm.config.hidden_size, hidden_size),
          nn.LeakyReLU(),
          nn.Dropout(dropout_rate),
        )
        self.linear = nn.Linear(hidden_size, n_layers)
        self.softmax = nn.Softmax(dim=1)
        print(f"n_layers: {n_layers}")

    def forward(self, _input, _label):

        # ipdb.set_trace()
        b_input = _input.view(1, _input.size(0))
        output = self.layers(b_input)
        logit = self.linear(output)
        prob = self.softmax(logit)
        return output, logit, prob



